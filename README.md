# Final Submission — Shipped + fp16

KLA AI Hackathon — AI-Based Restoration of Degraded Images (joint
denoising + 2x super-resolution on SEM images).

## Run it

```bash
python run.py <input_dir> <output_dir>
```

- `<input_dir>`: directory of degraded input images, one `.npy` file per
  image (single-channel float32, any H×W).
- `<output_dir>`: created if missing. Each output is written as
  `<same filename>.npy`, float32, values in `[0, 1]`, at 2× the input's
  spatial resolution.

No setup beyond `torch` and `numpy` — the model definition (`model.py`)
and trained weights (`weights/model_best.pth`) are bundled alongside
`run.py` and located relative to it, so it runs correctly from any
working directory.

Inference runs fp16 + channels_last + `torch.compile(mode="reduce-
overhead")` + batch=8 on GPU (falls back to fp32, batch=1, uncompiled on
CPU automatically). fp16 costs no measurable quality versus fp32 for this
model — verified on the full validation set.

### A note on `torch.compile`'s one-time cost

`torch.compile` pays a real, one-time compilation tax on the *first*
batch — several seconds, spent building and tuning a CUDA kernel for
this exact model and input shape. Every batch after that runs on the
compiled kernel and is meaningfully faster. That means whether compiling
is worth it at all depends on how many images you're processing in one
run:

| Images processed | fp16 + channels_last, no compile | + compile | Net effect |
|---|---|---|---|
| 200 | 37.7 img/s | 38.7 img/s | ~a wash — the compile tax barely pays for itself |
| 1,000 | 73.7 img/s | 106.6 img/s | **1.45× faster** — the tax is a small fraction of the total run |

(Measured on the same GPU, same batch size, same model — only
`torch.compile` toggled.) The larger the batch of images you hand to a
single `run.py` invocation, the more that one-time cost gets diluted
across steady-state batches, and the more compiling wins. This script
compiles unconditionally rather than trying to guess your dataset size
up front — on any reasonably sized evaluation set (hundreds of images or
more) it's a clear net win; on a very small one it costs a few seconds
you wouldn't otherwise spend. If `torch.compile` fails for any reason
(older PyTorch, unsupported GPU), `run.py` catches it automatically and
falls back to the uncompiled fp16 path rather than crashing.

## Architecture, in brief

**NAFNet-full**, bias-free convolutions, additive-residual joint
denoise + 2× super-resolution over a bicubic baseline:

1. The 128×128 noisy input is **bicubic-upsampled to 256×256** first —
   this upsampled image is both the only thing the network's trunk
   processes and the base the network's output is added onto.
2. A 4-level **encoder** (channel width 32, block counts 2/2/4/8) with
   skip connections, built from `NAFBlock`s (LayerNorm2d → 1×1/3×3/1×1
   convs → SimpleGate → simplified channel attention → residual).
3. **12 middle blocks** at the bottleneck.
4. A matching 4-level **decoder** (block counts 2/2/2/2), upsampling via
   `PixelShuffle(2)` at each stage, adding back the corresponding
   encoder skip connection.
5. Output = `clamp(bicubic_input + learned_residual, 0, 1)`.

**29.07M parameters.** Trained 600 epochs (~5.2h) on the project's noisy
LR / GT pairs. Final validation quality: **PSNR 23.502dB, SSIM 0.6317,
LPIPS 0.1426** — the best quality profile measured across every
architecture tried in this project (an alternative "Unified" architecture
trades some of this quality for substantially higher throughput; this
one was chosen for submission for its quality lead and simpler,
lower-risk stack).

## Files

```
model.py          -- self-contained model definition (torch only, no
                      external repo dependency)
run.py            -- inference script, see "Run it" above
weights/
  model_best.pth  -- trained checkpoint
```
