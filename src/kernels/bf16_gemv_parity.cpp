// src/kernels/bf16_gemv_parity.cpp - P2.S5's test for the BF16 GEMV.
//
// FOUR THINGS ARE CHECKED, and the third is the one the engine's correctness rests on:
//
//   1. both kernels against a host reference that reads the SAME bf16 bits - so what is under test is the
//      reduction, not the conversion, and the tolerance is set by summation order and nothing else;
//   2. the naive and the split kernel agree with each other, because the naive one is the reference the
//      split one exists to be checked against and a disagreement there means one of them is wrong;
//   3. **THE ACTIVATION CONTRACT IS OBSERVABLE.**  A BF16 weight sees a BF16 activation; feeding it fp16 or
//      f32 instead is a 2^-9 = 1.95e-03 relative change, which is above any 1e-3 tolerance.  An fp16
//      activation is the rival reading, computed and required to differ, because it is what the engine did
//      before `docs/activation-contract.md` settled the question and it produces perfectly plausible output.
//   4. **THE MULTI-ROW WINDOW IS ROW-COUNT INDEPENDENT.**  `bf16_gemv_fp32_mmvf_multi` dispatches on NT
//      instantiations; while the window was 8 rows the largest instantiation was 8 and a wider launch would
//      have silently computed fewer rows than it was asked for.  Every width 1..kVerifyMaxT, every row compared
//      BITWISE against that row's own single-row call.  (`bf16_gemv_parity --selftest` runs it; the rows path of
//      the _aux entry needs STRATA_MMVF_ROWS in the process environment - see section 4 below.)
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The host reference, in DOUBLE, over the same bf16 bits.
void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(x[(size_t) i]) * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: bf16_gemv_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;

    // THE REAL SHAPES, because a kernel that only works on a square toy is not a kernel this engine can use.
    // `ssm_alpha` is [2560, 48] - a wide reduction and 48 outputs - and `indexer.k_proj` is [2560, 128].
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        // a realistic weight scale: these are projections, not embeddings
        for (auto& v : fw) v = g(rng) * 0.05f;

        // THE ACTIVATION IS ROUNDED TO BF16 FIRST, which is the contract, and the WEIGHT is already bf16 in
        // the pack - the loaded bytes ARE the high 16 bits of an f32.
        std::vector<uint16_t> x((size_t) s.n_in), w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);

        uint16_t *d_x = nullptr, *d_w = nullptr;
        float *d_y = nullptr;
        check(cudaMalloc(&d_x, x.size() * 2), "x");
        check(cudaMalloc(&d_w, w.size() * 2), "w");
        check(cudaMalloc(&d_y, (size_t) s.n_out * 4), "y");
        check(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "cx");
        check(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "cw");

        std::vector<float> naive((size_t) s.n_out), warp((size_t) s.n_out), split((size_t) s.n_out);
        strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
        check(cudaMemcpy(naive.data(), d_y, naive.size() * 4, cudaMemcpyDeviceToHost), "cy1");
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 32, nullptr);
        check(cudaMemcpy(warp.data(), d_y, warp.size() * 4, cudaMemcpyDeviceToHost), "cy2");
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 256, nullptr);
        check(cudaMemcpy(split.data(), d_y, split.size() * 4, cudaMemcpyDeviceToHost), "cy3");

        const double rn = rel_l1(want, naive), rw = rel_l1(want, warp), rs = rel_l1(want, split);
        std::printf("  %-34s naive %.2e  warp %.2e  split256 %.2e\n", s.what, rn, rw, rs);
        // The products are EXACT - both sides are bf16-valued, so f32 represents each product exactly - and
        // the only difference is the order the 2560 terms are added.  A tight bound is therefore meaningful.
        if (!(rn <= 1e-5 && rw <= 1e-5 && rs <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 3. THE ACTIVATION CONTRACT.  The rival: an fp16 activation instead of bf16, which is what the
        // engine did before the contract was settled.  It must be observable, or check 1 above would pass
        // against either and the contract would be untested.
        {
            std::vector<float> fp16_as_f32((size_t) s.n_in);
            std::vector<uint16_t> x16((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) {
                x16[i] = strata::kernels::f16_from_f32(fx[i]);
                fp16_as_f32[i] = strata::kernels::f32_from_f16(x16[i]);
            }
            // Round the fp16 value to bf16 - i.e. give the kernel a bf16 activation whose VALUES came through
            // fp16.  That is the whole difference between the two contracts at this call site.
            for (size_t i = 0; i < fp16_as_f32.size(); ++i) x[i] = bf16_from_f32(fp16_as_f32[i]);
            check(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "cx2");
            std::vector<float> rival((size_t) s.n_out);
            strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
            check(cudaMemcpy(rival.data(), d_y, rival.size() * 4, cudaMemcpyDeviceToHost), "cy4");
            const double r = rel_l1(want, rival);
            const bool visible = r > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "bf16 vs fp16 activation",
                        visible ? "yes" : "*** NO ***", r * 100);
            if (!visible) ++bad;
        }
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
    }

    // ---- 4. THE MULTI-ROW KERNEL IS ROW-COUNT INDEPENDENT.  bf16_gemv_fp32_mmvf_multi (native_bf16.cu's NT
    // dispatch) instantiated NT up to the window bound; the contract in bf16_gemv.hpp is "every output
    // bit-identical to its own single-row call", n_tok = 1..kVerifyMaxT.  So: every width 1..kVerifyMaxT, and
    // EVERY row of EVERY width compared BITWISE (memcmp on the float bits, no tolerance) against that row's own
    // bf16_gemv_fp32_mmvf call.  The output is poisoned with NaN bits before each launch, so a row the kernel
    // did not write cannot match.
    //
    // THE ROWS PATH OF THE _aux ENTRY IS REPORTED, NOT FORCED: it arms itself only when STRATA_MMVF_ROWS is in
    // the PROCESS environment (the harness does not set its own environment - running the test with the variable
    // set covers it); when multi_aux declines there is nothing launched and this stays UNCOVERED here.
    {
        const int MT = strata::kernels::kVerifyMaxT;   // the window's rows: the loop and every buffer follow the constant
        struct MShape { long long n_in, n_out; const char* what; };
        const MShape mshapes[] = {
            {2560, 48, "multi ssm_alpha  [2560, 48]"}, {2560, 128, "multi k_proj  [2560, 128]"},
            {128, 16, "multi small       [128, 16]"}, {64, 4, "multi small          [64, 4]"},
        };
        for (const MShape& m : mshapes) {
            std::mt19937 rng(9091);
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> fx((size_t) MT * m.n_in);
            for (auto& v : fx) v = g(rng);
            std::vector<uint16_t> w((size_t) (m.n_in * m.n_out)), w_aux((size_t) m.n_in);
            for (auto& v : w) v = bf16_from_f32(g(rng) * 0.05f);
            for (auto& v : w_aux) v = bf16_from_f32(g(rng) * 0.05f);
            float *d_x = nullptr, *d_y = nullptr, *d_ya = nullptr, *d_s = nullptr;
            uint16_t *d_w = nullptr, *d_wa = nullptr;
            check(cudaMalloc(&d_x, fx.size() * sizeof(float)), "mx");
            check(cudaMalloc(&d_w, w.size() * sizeof(uint16_t)), "mw");
            check(cudaMalloc(&d_wa, w_aux.size() * sizeof(uint16_t)), "mwa");
            check(cudaMalloc(&d_y, (size_t) MT * m.n_out * sizeof(float)), "my");
            check(cudaMalloc(&d_ya, (size_t) MT * sizeof(float)), "mya");
            check(cudaMalloc(&d_s, (size_t) m.n_out * sizeof(float)), "ms");
            check(cudaMemcpy(d_x, fx.data(), fx.size() * sizeof(float), cudaMemcpyHostToDevice), "mcx");
            check(cudaMemcpy(d_w, w.data(), w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "mcw");
            check(cudaMemcpy(d_wa, w_aux.data(), w_aux.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "mcwa");

            // THE PER-ROW SINGLE-ROW REFERENCE: row t of an NT-wide result must be THESE bits for every NT >= t.
            std::vector<std::vector<float>> ref((size_t) MT, std::vector<float>((size_t) m.n_out));
            std::vector<float> ref_aux((size_t) MT, 0.f);
            for (int t = 0; t < MT; ++t) {
                check(cudaMemset(d_s, 0x5a, (size_t) m.n_out * 4), "ms poison");
                strata::kernels::bf16_gemv_fp32_mmvf(d_x + (size_t) t * m.n_in, d_w, d_s, m.n_in, m.n_out, nullptr);
                check(cudaMemcpy(ref[(size_t) t].data(), d_s, ref[(size_t) t].size() * 4, cudaMemcpyDeviceToHost), "cref");
                check(cudaMemset(d_s, 0x5a, 4), "ms poison aux");
                strata::kernels::bf16_gemv_fp32_mmvf(d_x + (size_t) t * m.n_in, d_wa, d_s, m.n_in, 1, nullptr);
                check(cudaMemcpy(&ref_aux[(size_t) t], d_s, 4, cudaMemcpyDeviceToHost), "cref_aux");
            }

            int rows_diff = 0, sets = 0;
            for (int NT = 1; NT <= MT; ++NT) {
                check(cudaMemset(d_y, 0xFF, (size_t) MT * m.n_out * 4), "my poison");
                strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, m.n_in, d_w, d_y, m.n_out, m.n_in, m.n_out, NT,
                                                           nullptr);
                std::vector<float> got((size_t) NT * m.n_out);
                check(cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost), "cmulti");
                for (int t = 0; t < NT; ++t) {
                    ++sets;
                    if (std::memcmp(&got[(size_t) t * m.n_out], ref[(size_t) t].data(), (size_t) m.n_out * 4) != 0) {
                        std::printf("    *** %s: multi NT=%d row %d differs from its single-row call ***\n", m.what, NT,
                                    t);
                        ++rows_diff;
                    }
                }
            }

            // the _aux rows path, armed or not by the CALLER's environment (STRATA_MMVF_ROWS; declined = uncovered)
            int aux_diff = 0;
            const char* aux_note;
            check(cudaMemset(d_y, 0xFF, (size_t) MT * m.n_out * 4), "my poison aux");
            check(cudaMemset(d_ya, 0xFF, (size_t) MT * 4), "mya poison aux");
            if (!strata::kernels::bf16_gemv_fp32_mmvf_multi_aux(d_x, m.n_in, d_w, d_y, m.n_out, m.n_in, m.n_out, MT,
                                                                d_wa, d_ya, 1, nullptr)) {
                aux_note = "rows path UNCOVERED (STRATA_MMVF_ROWS not set in this environment)";
            } else {
                aux_note = "rows path ARMED: aux outputs compared bitwise";
                std::vector<float> got((size_t) MT * m.n_out);
                check(cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost), "caux_main");
                std::vector<float> gaux((size_t) MT);
                check(cudaMemcpy(gaux.data(), d_ya, (size_t) MT * 4, cudaMemcpyDeviceToHost), "caux");
                for (int t = 0; t < MT; ++t) {
                    if (std::memcmp(&got[(size_t) t * m.n_out], ref[(size_t) t].data(), (size_t) m.n_out * 4) != 0)
                        ++aux_diff;
                    if (std::memcmp(&gaux[(size_t) t], &ref_aux[(size_t) t], 4) != 0) ++aux_diff;
                }
            }
            std::printf("  %-34s rows 1..%d: %d of %d row compares differ; %s\n", m.what, MT, rows_diff + aux_diff,
                        sets, aux_note);
            if (rows_diff || aux_diff) ++bad;
            cudaFree(d_x); cudaFree(d_w); cudaFree(d_wa); cudaFree(d_y); cudaFree(d_ya); cudaFree(d_s);
        }
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}
