// attn_bench: qsa_decode_attn_batch on production-like geometry (int8 K/V, page size 4, shuffled page table).
//   attn_bench [reps]   - per case: us per call and an FNV hash of the output (variants via STRATA_ATTN_* env)
// main: 2,051 selected cells of a 32K context (the main model's QSA attention); mtp: 3,500 live cells, dense ids,
// a 32K-cell capacity (the MTP draft layer's attention); each for 1 and 4 queries.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
using namespace strata::kernels;
template <class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc((void**) &d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    QsaShapes s = qsa_real_shapes();
    const int64_t NKV = s.n_head_kv, HD = s.head_dim, NH = s.n_head, ps = s.page_size;
    const int64_t CELLS = 32768, pages = CELLS / ps;
    std::mt19937 rng(7);
    std::vector<int32_t> table((size_t) pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<int8_t> kq((size_t) pages * NKV * ps * HD), vq(kq.size());
    for (auto& x : kq) x = (int8_t) (rng() % 255 - 127);
    for (auto& x : vq) x = (int8_t) (rng() % 255 - 127);
    std::vector<uint16_t> ksc(kq.size() / 64), vsc(kq.size() / 64);
    for (auto& x : ksc) x = (uint16_t) (0x2000 + rng() % 0x400);   // ~0.008 fp16
    for (auto& x : vsc) x = (uint16_t) (0x2000 + rng() % 0x400);
    QsaAttnPools pools;
    pools.k_q = up(kq); pools.v_q = up(vq); pools.k_scale = up(ksc); pools.v_scale = up(vsc);
    pools.page_table = up(table);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Case { const char* name; int64_t cap, live; bool dense; };
    const Case cases[] = {{"main 2051 of 32K", 2112, 2051, false}, {"mtp 3500 dense", CELLS, 3500, true}};
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (const Case& c : cases)
        for (int nq : {1, 4}) {
            std::vector<int32_t> ids((size_t) nq * c.cap, 0), steps((size_t) nq * kStepCount, 0);
            for (int q = 0; q < nq; ++q) {
                std::vector<int32_t> sel;
                if (c.dense) { sel.resize((size_t) c.live); std::iota(sel.begin(), sel.end(), 0); }
                else {
                    std::vector<int32_t> all(CELLS); std::iota(all.begin(), all.end(), 0);
                    std::shuffle(all.begin(), all.end(), rng);
                    sel.assign(all.begin(), all.begin() + c.live); std::sort(sel.begin(), sel.end());
                }
                std::copy(sel.begin(), sel.end(), ids.begin() + (size_t) q * c.cap);
                if (c.dense) for (int64_t i = c.live; i < c.cap; ++i) ids[(size_t) (q * c.cap + i)] = (int32_t) i;
                steps[(size_t) q * kStepCount + kStepNKv] = c.dense ? (int32_t) c.live : (int32_t) CELLS;
                steps[(size_t) q * kStepCount + kStepWidth] = (int32_t) c.live;
            }
            std::vector<float> qv((size_t) nq * NH * HD);
            for (auto& x : qv) x = nd(rng);
            int32_t* dids = up(ids); int32_t* dst = up(steps); float* dq = up(qv);
            float *scr = nullptr, *att = nullptr;
            cudaMalloc((void**) &scr, (size_t) nq * qsa_decode_attn_scratch_floats(c.cap, s) * sizeof(float));
            cudaMalloc((void**) &att, (size_t) nq * NH * HD * sizeof(float));
            qsa_decode_attn_batch(dq, pools, dids, dst, c.cap, s, scr, att, nq, nullptr);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) qsa_decode_attn_batch(dq, pools, dids, dst, c.cap, s, scr, att, nq, nullptr);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            std::vector<float> out((size_t) nq * NH * HD);
            cudaMemcpy(out.data(), att, out.size() * 4, cudaMemcpyDeviceToHost);
            uint64_t h = 1469598103934665603ull;
            for (float f : out) { uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ull; }
            std::printf("%-18s nq %d: %7.1f us  hash %016llx\n", c.name, nq, 1000.0 * ms / reps, (unsigned long long) h);
            cudaFree(dids); cudaFree(dst); cudaFree(dq); cudaFree(scr); cudaFree(att);
        }
    return 0;
}
