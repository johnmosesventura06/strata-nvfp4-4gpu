"""tools/ple_bf16_pack.py - the PLE n-gram table in BF16 (the checkpoint's own full-precision form), as a GGUF.

    python tools/ple_bf16_pack.py --bin <ngram_embedding.weight.bin> --out <ple-bf16.gguf> [--rows 320001536 --dim 160]

Inputs a raw row-major [rows, dim] BF16 monolith - exactly the tensor a checkpoint stores (golden lane's
`language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight.bin` is one, 102,400,491,520 bytes).
The bytes go through untouched: this writes the GGUF envelope and nothing else, so precision is what the source
tensor was, by construction.

The engine reads `per_layer_token_embd.weight` (ne = [dim, rows], type BF16 = 30); BF16 is self-describing:
no `strata.ple.format` marker, no scale (ngram.cpp's format list). Alone in the file, the table must fill it
exactly from the aligned data start - the engine checks that arithmetic before mapping ("PLE table size mismatch").
"""
import argparse, json, pathlib, struct, sys

GGML_TYPE_BF16 = 30
ROWS_DEFAULT, DIM_DEFAULT = 320001536, 160
ALIGN, CHUNK = 32, 64 << 20


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", 8) + gguf_string(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True, help="raw row-major [rows, dim] BF16 tensor file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=ROWS_DEFAULT)
    ap.add_argument("--dim", type=int, default=DIM_DEFAULT)
    ap.add_argument("--name", default="", help="optional model-dir name for the provenance key")
    a = ap.parse_args()
    src, out = pathlib.Path(a.bin), pathlib.Path(a.out)

    nbytes = src.stat().st_size
    if nbytes != a.rows * a.dim * 2:
        sys.exit("size mismatch: %s holds %d bytes, %d rows x %d BF16 needs %d"
                 % (src, nbytes, a.rows, a.dim, a.rows * a.dim * 2))

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, BF16 as shipped"),
           kv_string("strata.ple.source", a.name or src.parent.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", a.dim, a.rows)
    head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    written = 0
    with open(tmp, "wb") as w:
        w.write(head)
        with open(src, "rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                w.write(b)
                written += len(b)
                if written % (16 << 30) < CHUNK:
                    print("  %.1f GB" % (written / 1e9), flush=True)
    if written != nbytes:
        sys.exit("short copy: %d of %d" % (written, nbytes))
    if tmp.stat().st_size != len(head) + nbytes:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + nbytes))
    tmp.replace(out)
    print("wrote %s: %d rows x %d BF16 (%.1f GB), header %d B" % (out, a.rows, a.dim, nbytes / 1e9, len(head)))


if __name__ == "__main__":
    main()
