// include/strata/core/split_head.hpp - the verify window's output head across the peer GPUs.
//
// Ported from the four-GPU GGUF engine (johnmosesventura06/strata-4gpu, src/core/split_head.*, the
// parity-proven N-way head split) for the NVFP4 lane, owner GO 2026-10-11; docs/HEAD-SPLIT-PORT.md is
// the design record.  The head streams the whole output matrix after the last layer (this pack's
// output.weight is Q8_0 [2560, 248320] = 675,430,400 bytes): the main GPU computes rows [0, split) and
// each peer holds [split_i, end_i), so every card streams part of it.  The main GPU's window graph
// writes the head's input rows (the final hyper-connection mix, T x n_in floats) to mapped memory and
// raises a count; the host then launches each peer's part (the same quantization, projection and
// per-row argmax kernels, so every logit is the one-GPU head's), which writes each token's pick and
// logit to mapped memory and raises a count of its own.  The host keeps the larger logit, on equality
// the main GPU's (the lower index).  With `logits`, a window's rows are copied to mapped memory too,
// for sampled requests.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class SplitHead {
public:
    SplitHead() = default;
    ~SplitHead();
    SplitHead(const SplitHead&) = delete;
    SplitHead& operator=(const SplitHead&) = delete;

    /// Rows [split, end) of `output.weight` in the model's shards on `device`, for windows of up to `max_t` tokens
    /// (`end` <= 0 means n_vocab).  `main_device` is current again after every call.
    bool init(int device, int main_device, const std::vector<std::string>& shards, int64_t n_in, int64_t n_vocab,
              int64_t split, int max_t, std::string& err, int64_t end = -1);
    bool on() const { return dev_ >= 0; }
    int64_t split() const { return split_; }
    int64_t end() const { return end_; }
    int64_t part_rows() const { return rows_; }
    uint64_t bytes() const { return bytes_; }
    /// GPU-side protocol (phase 2.5): the verifier passes its assembled full-vocab buffer once its buffers bind.
    /// With P2P to the main device, the part's window graph writes its logits rows straight into
    /// `head_full[t][split..end)` (no mapped logits, no main-side join for this part) and pre-captures every
    /// (T, logits) graph, which then starts with a wait-value on hand_dev() so the host can launch it at the
    /// window's start instead of polling the handoff.  Precedence: call before any submit().
    void set_target(float* head_full, int64_t n_vocab);
    /// The part's handoff flag in ITS OWN device memory: the main's window graph bumps it (a one-word P2P write,
    /// ordered after the mapped input rows), the part's graph wait-values on it and zeroes it in-graph.
    uint32_t* hand_dev() const { return hand_dev_; }
    bool gpu_protocol() const { return p2p_ok_; }
    /// Mapped: the head's input rows the main GPU writes (max_t x n_in floats) and the count it raises after them.
    float* input() const { return h_in_; }
    uint32_t* handoff() const { return h_hand_; }
    /// Launches this GPU's part for a window of T tokens.  gpu_protocol(): the graph waits on hand_dev() itself -
    /// launch at the window's start, no host spin.  Otherwise: call after the handoff count has risen.
    bool submit(int T, bool logits, std::string& err);
    /// Until this GPU's count reaches `n` (the submits so far).
    bool wait(uint32_t n, std::string& err);
    /// Per token: the pick's row within this part, its logit; with the mapped protocol, T x (end - split) logits
    /// after a `logits` submit (the gpu protocol writes them into the target instead).
    const int32_t* idx() const { return h_idx_; }
    const float* val() const { return h_val_; }
    const float* logits() const { return h_logits_; }

private:
    bool capture(int T, bool logits, std::string& err);

    int dev_ = -1, main_ = 0, max_t_ = 0, type_ = -1;
    int64_t n_in_ = 0, split_ = 0, end_ = 0, rows_ = 0;
    uint64_t bytes_ = 0;
    cudaStream_t s_ = nullptr;
    void* w_ = nullptr;                                         // the rows' GGUF blocks
    float *mixed_ = nullptr, *out_ = nullptr;                   // device: the input, the logits
    uint8_t *xq_ = nullptr, *xil_ = nullptr, *scratch_ = nullptr;
    uint32_t* hand_dev_ = nullptr;                              // device: the wait-value handoff flag
    float* target_ = nullptr;                                   // the verifier's head_full_ (P2P logits rows)
    int64_t target_n_ = 0;
    bool p2p_ok_ = false;
    float* h_in_ = nullptr;                                     // mapped (portable)
    uint32_t *h_hand_ = nullptr, *h_done_ = nullptr;
    int32_t* h_idx_ = nullptr;
    float *h_val_ = nullptr, *h_logits_ = nullptr;
    cudaGraphExec_t exec_[9][2] = {};
};

}  // namespace strata::core
