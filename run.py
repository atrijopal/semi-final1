#!/usr/bin/env python3
"""
Standalone evaluation / inference script for the KLA AI Hackathon --
AI-Based Restoration of Degraded Images (joint denoising + 2x super-
resolution, SEM images).

Usage:
    python run.py <input_dir> <output_dir>

<input_dir>  : directory of degraded input images, one .npy file per image
               (single-channel float32 array, any H x W -- the model
               upsamples 2x, so a 128x128 input produces a 256x256 output).
<output_dir> : directory to write restored images to (created if missing).
               Each output is written as <same filename>.npy, float32,
               values in [0, 1], at 2x the input's spatial resolution.

No manual edits required -- the model architecture (model.py) and trained
weights (weights/model_best.pth) are bundled alongside this script and
located via a path relative to this file, so it runs correctly regardless
of the working directory it's invoked from.

Model: "Shipped" (plain NAFNet-full, bicubic-first trunk, PixelShuffle
decoder, 29.07M params) -- see README.md for the architecture brief.
Inference runs at fp16 + channels_last + torch.compile(mode=
"reduce-overhead") + batch=8 on GPU (zero measured quality cost from fp16
alone; compile adds a one-time warmup cost on the first batch in exchange
for materially higher steady-state throughput). Falls back to fp32,
batch=1, uncompiled on CPU. If torch.compile fails for any reason (older
PyTorch, unsupported GPU, etc.) this script catches it and automatically
falls back to the uncompiled fp16 model rather than crashing. Inputs are
grouped into batches of up to 8, split whenever resolution changes, so
mixed-resolution folders still work correctly -- this project's own data
is uniformly 128x128.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import build_model  # noqa: E402

CHECKPOINT_PATH = os.path.join(HERE, "weights", "model_best.pth")
BATCH_SIZE = 8


def bicubic_upsample(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """x: (N,1,H,W) tensor, any value range (kept as-is, not clipped --
    the degraded input's out-of-[0,1] range is expected). Returns
    (N,1,2H,2W)."""
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h * scale, w * scale), mode="bicubic", align_corners=False)


def load_batches(files, input_dir, batch_size):
    """Yields (filenames, stacked_array) groups of up to `batch_size`
    files, starting a new group whenever the array shape changes (so a
    folder of mixed resolutions is still handled correctly, just without
    batching across the boundary)."""
    names, arrs = [], []
    for fname in files:
        arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
        if arr.ndim == 3:  # tolerate an (H,W,1)-style array defensively
            arr = arr[..., 0]
        if arrs and (arr.shape != arrs[-1].shape or len(arrs) >= batch_size):
            yield names, np.stack(arrs)
            names, arrs = [], []
        names.append(fname)
        arrs.append(arr)
    if arrs:
        yield names, np.stack(arrs)


def main():
    ap = argparse.ArgumentParser(description="Restore degraded SEM images: joint denoise + 2x super-resolution.")
    ap.add_argument("input_dir", help="Directory of degraded input .npy images")
    ap.add_argument("output_dir", help="Directory to write restored .npy images to")
    args = ap.parse_args()

    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    model = build_model()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).to(dtype).eval()
    model_eager = model  # kept as an uncompiled fallback if torch.compile fails
    if use_fp16:
        model = model.to(memory_format=torch.channels_last)
        model_eager = model
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[run] WARNING: torch.compile unavailable ({e}); "
                  f"continuing uncompiled (fp16 + channels_last only).", file=sys.stderr)
            model = model_eager
        torch.backends.cudnn.benchmark = True

    t_model_ready = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not input_files:
        print(f"[run] WARNING: no .npy files found in {args.input_dir}", file=sys.stderr)

    # torch.compile(mode="reduce-overhead") uses CUDA Graphs, which only pay
    # off when the input tensor sits at a FIXED GPU memory address on every
    # call -- a fresh tensor built from newly-loaded numpy data each batch
    # (a different address every time) defeats that, and can even make the
    # graph-managed path slower than plain eager. So: one persistent input
    # buffer, sized to the first full-size batch seen, refreshed in place
    # via copy_() each iteration -- that's what actually lets the compiled
    # graph replay fast. A batch that doesn't match that buffer's shape
    # (a short final batch, or a different resolution) runs on the eager
    # model instead, since CUDA Graphs can't handle a shape change either.
    static_buffer = None
    fell_back = False
    n_written = 0
    with torch.no_grad():
        for names, arr in load_batches(input_files, args.input_dir, BATCH_SIZE):
            lr = torch.from_numpy(arr).unsqueeze(1).to(device)  # (B,1,H,W)
            bicubic = bicubic_upsample(lr, scale=2).to(dtype)

            use_compiled = (
                model is not model_eager and not fell_back
                and bicubic.shape[0] == BATCH_SIZE
            )
            if use_compiled:
                if static_buffer is None or static_buffer.shape != bicubic.shape:
                    static_buffer = torch.empty_like(bicubic).to(memory_format=torch.channels_last)
                static_buffer.copy_(bicubic)
                try:
                    out = model(static_buffer)
                except Exception as e:
                    print(f"[run] WARNING: compiled forward pass failed ({e}); "
                          f"falling back to uncompiled fp16 for the rest of the run.", file=sys.stderr)
                    fell_back = True
                    bicubic_e = bicubic.to(memory_format=torch.channels_last) if use_fp16 else bicubic
                    out = model_eager(bicubic_e)
            else:
                if use_fp16:
                    bicubic = bicubic.to(memory_format=torch.channels_last)
                out = model_eager(bicubic)

            out_np = out.float().cpu().numpy()
            for i, fname in enumerate(names):
                np.save(os.path.join(args.output_dir, fname), out_np[i, 0])
            n_written += len(names)

    t_end = time.time()

    print(f"[run] device={device}, precision={'fp16' if use_fp16 else 'fp32'}")
    print(f"[run] model init: {t_model_ready - t_start:.3f}s")
    print(f"[run] inference + I/O for {n_written} images: {t_end - t_model_ready:.3f}s "
          f"({(t_end - t_model_ready) / max(1, n_written) * 1000:.2f} ms/image)")
    print(f"[run] total wall-clock: {t_end - t_start:.3f}s")
    print(f"[run] wrote {n_written} restored images to {args.output_dir}")


if __name__ == "__main__":
    main()
