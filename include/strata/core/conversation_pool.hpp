#pragma once

// Host RAM for conversation images (ConversationBuffer segments) and the journal's 16 MiB I/O blocks, kept OUT of
// the malloc heap.  Measured 10-19: glibc served the freed 16 MiB segments from its heap (its dynamic mmap
// threshold rises to any freed mmapped chunk up to 32 MiB) and never gave the freed middle of the heap back, so a
// day of parks and mounts (1,350 + 985 images, up to 5.1 GiB each) left 6.7 GiB of free-but-held heap: available
// RAM 16 -> 8.6 GiB, back to 15.6 after a malloc_trim.  Here:
//   - a FIXED pool of 16 MiB blocks (STRATA_CONV_POOL_MIB, default 4096; 0 = no pool), reserved and touched up
//     front by reserve() so its RAM is paid once, like the pinned K/V, and reused by every park and mount;
//     Any request from 1 MiB to 16 MiB takes a whole block (a per-layer K/V buffer is usually a little under
//     16 MiB; mapping those fresh made an 18K-token park 150 -> 780 ms on page faults);
//   - a block the pool has no spare for, and anything above 16 MiB, mapped from the kernel and unmapped on
//     free - an image larger than the pool costs RAM only while it exists;
//   - smaller allocations stay on operator new.
// So the process holds the pool plus whatever images are alive this moment, never a high-water mark.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace strata::core::conv_pool {

constexpr size_t kBlock = 16u << 20;
constexpr size_t kDirectMin = 1u << 20;
constexpr size_t kPage = 4096;

struct Stats {
    size_t cap_blocks = 0;         // the pool's size in 16 MiB blocks
    size_t free_blocks = 0;        // idle blocks held right now (never more than cap_blocks)
    uint64_t pool_hits = 0, pool_misses = 0, mapped = 0, unmapped = 0;
};

struct State {
    std::mutex mu;
    std::vector<void*> free;
    size_t cap = 0;
    bool configured = false;
    Stats st;
};
inline State& state() {
    static State* s = new State();   // never destroyed: blocks may be released by static destructors elsewhere
    return *s;
}
inline size_t env_cap_blocks() {
    const char* e = std::getenv("STRATA_CONV_POOL_MIB");
    const long long mib = e ? std::atoll(e) : 4096;
    return mib > 0 ? (size_t) mib / 16 : 0;
}
inline void configure_locked(State& s) {
    if (!s.configured) { s.cap = env_cap_blocks(); s.configured = true; }
}

#ifndef _WIN32
inline void* map_bytes(size_t n) {
    void* p = ::mmap(nullptr, (n + kPage - 1) / kPage * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline void unmap_bytes(void* p, size_t n) { ::munmap(p, (n + kPage - 1) / kPage * kPage); }
#endif

/// Pre-reserves the whole pool and touches every page (MAP_POPULATE).  Returns the bytes now held.
inline size_t reserve() {
#ifdef _WIN32
    return 0;
#else
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    configure_locked(s);
    while (s.free.size() < s.cap) {
        void* p = ::mmap(nullptr, kBlock, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p == MAP_FAILED) break;
        s.free.push_back(p);
    }
    return s.free.size() * kBlock;
#endif
}

/// Page-aligned for n >= kDirectMin.  Throws std::bad_alloc like operator new.
inline void* alloc(size_t n) {
#ifndef _WIN32
    if (n >= kDirectMin) {
        State& s = state();
        if (n <= kBlock) {
            std::lock_guard<std::mutex> lk(s.mu);
            configure_locked(s);
            if (!s.free.empty()) {
                void* p = s.free.back();
                s.free.pop_back();
                ++s.st.pool_hits;
                return p;
            }
            ++s.st.pool_misses;
        }
        if (void* p = map_bytes(n <= kBlock ? kBlock : n)) {   // block-sized, so it can join the pool on free
            std::lock_guard<std::mutex> lk(s.mu);
            ++s.st.mapped;
            return p;
        }
        throw std::bad_alloc();
    }
#endif
    return ::operator new(n);
}

inline void release(void* p, size_t n) noexcept {
    if (!p) return;
#ifndef _WIN32
    if (n >= kDirectMin) {
        State& s = state();
        {
            std::lock_guard<std::mutex> lk(s.mu);
            configure_locked(s);
            if (n <= kBlock && s.free.size() < s.cap) {   // idle blocks never exceed the pool's size
                s.free.push_back(p);
                return;
            }
            ++s.st.unmapped;
        }
        unmap_bytes(p, n <= kBlock ? kBlock : n);
        return;
    }
#endif
    ::operator delete(p);
}

inline Stats stats() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    Stats out = s.st;
    out.cap_blocks = s.cap;
    out.free_blocks = s.free.size();
    return out;
}

/// std::allocator replacement for the segments: routes through alloc()/release().
template <class T> struct Allocator {
    using value_type = T;
    Allocator() noexcept = default;
    template <class U> Allocator(const Allocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(alloc(n * sizeof(T))); }
    void deallocate(T* p, size_t n) noexcept { release(p, n * sizeof(T)); }
    template <class U> bool operator==(const Allocator<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const Allocator<U>&) const noexcept { return false; }
};

}  // namespace strata::core::conv_pool
