# ANIME 4K UPSCALER

Paste an anime clip → get an **Ultra 4K UHD** version, upscaled by AI on
**your laptop's GPU**. The audio track is **copied bit-for-bit** — it sounds
exactly the same as the original.

- Works on **any laptop GPU**: NVIDIA, AMD, or Intel (including integrated
  graphics) — it uses Vulkan, so there's no CUDA/PyTorch to install.
- **No pip packages**: one Python file. On first run it auto-downloads the
  AI engine (Real-ESRGAN) and — on Windows and Linux — ffmpeg too
  (macOS: `brew install ffmpeg` once).
- **Built for speed**: anime-specific fast model (`realesr-animevideov3`),
  smart scale factor (1080p only needs 2×, not 4×), hardware video encoding
  (NVENC / QuickSync / AMF auto-detected), and the decode → GPU-upscale →
  encode stages run **in parallel**.
- **Frame-exact**: the source is decoded exactly once into a raw frame
  stream — there is no seeking, so no dropped or duplicated frames, even
  for trimmed/remuxed/VFR sources.

## Quick start

**Windows (easiest):** drag your clip onto **`Upscale to 4K.bat`**.
Or double-click it and paste the clip's path when asked.

**Any OS** (on macOS/Linux type `python3` instead of `python`):

```bash
python upscale.py                    # prompts you to paste the clip path
python upscale.py "my clip.mp4"      # direct
python upscale.py clip.mp4 --fast    # ~2x faster, still looks great
```

The result appears next to your clip as `my clip_4K.mp4`.

> Requirements: Python 3.8+ ([python.org](https://www.python.org/downloads/) —
> tick *"Add python.exe to PATH"*), an up-to-date GPU driver, and ~10 GB of
> free temp disk space (or use `--fast`, which needs ~10× less).

## How it works

```
clip.mp4 ─┬─ video ─▶ decode ONCE ─▶ Real-ESRGAN (GPU, Vulkan) ─▶ fit to 3840×2160 ─▶ encode ─┐
          │             (raw frame stream, chunked; all three stages run in parallel)         ├─▶ clip_4K.mp4
          └─ audio ───────────────────── copied unchanged (bit-for-bit) ─────────────────────┘
```

The video is decoded in a single pass into a raw frame stream that is
sliced into chunks by frame count (seam-proof by construction). While the
GPU upscales chunk *N*, the decoder is already producing chunk *N+1* and
the encoder is compressing chunk *N−1*. Temp disk stays bounded because
the decoder is blocked whenever it gets too far ahead.

The scale factor is picked automatically — the smallest one that reaches 4K
(1080p → 2×, 720p → 3×, ≤540p → 4×), because upscaling less is *much*
faster and any small remainder is finished with a high-quality Lanczos
resize. Anamorphic (non-square-pixel) and rotated sources are detected and
handled correctly, and every upscale is verified frame-by-frame — if the
GPU runs out of memory the tool retries automatically with smaller tiles.

## Tuned for: Alienware m16 R2 (RTX 4060 Laptop 8 GB + Core Ultra 9 185H)

The defaults are already set for this machine — just run it plain:

- The AI engine auto-picks the **RTX 4060** (discrete GPUs are preferred
  over the Intel Arc iGPU). If Task Manager ever shows GPU 0 busy and
  GPU 1 idle during a run, add `--gpu 1`.
- Encoding auto-selects **NVENC** (hardware HEVC) — the encode step is
  effectively free.
- 8 GB VRAM handles full-frame tiles at every scale — no `--tile` needed.
- The Intel **NPU (AI Boost) is not used** — this pipeline runs on Vulkan
  GPU compute, which the NPU doesn't expose. The 4060 is far faster anyway.
- Ballpark speed for 1080p → 4K: roughly **10–20 frames/s** (a 1-minute
  24 fps clip in ~1.5–2.5 minutes); `--fast` roughly doubles it.
  Plug in the charger — on battery the GPU throttles hard.

## Making it faster

| What | How | Effect |
|---|---|---|
| Fast mode | `--fast` | JPEG intermediates + faster encoder preset; ~2× faster, ~10× less temp disk |
| Force the discrete GPU | `--gpu 1` | Dual-GPU laptops sometimes default to the slow integrated GPU |
| More GPU threads | `--jobs 2:4:2` | Helps on beefy GPUs (load:proc:save threads) |
| Lower the target quality | `--quality 22` | Smaller/faster encode (CRF/CQ, lower = better) |
| Plug in the charger | — | Laptops heavily throttle the GPU on battery |

## All options

```
python upscale.py INPUT
  -o, --output FILE   output file (default: <name>_4K.mp4 next to the input)
  --fast              speed mode (jpeg intermediates + faster encode preset)
  --scale {auto,2,3,4}  AI scale factor (default: auto)
  --model {animevideo,anime-sharp,photo}
                      animevideo = fast anime video model (default)
                      anime-sharp = crisper lines, ~4x slower
  --encoder NAME      force an ffmpeg encoder (default: auto-detect hardware)
  --quality N         CRF/CQ quality, lower = better (default 16-19)
  --chunk N           frames per chunk (default 150; bounds temp disk usage)
  --tile N            GPU tile size (try 256 or 128 if you run out of VRAM)
  --gpu N             GPU index (dual-GPU laptops: try 1)
  --jobs L:P:S        upscaler threads as load:proc:save (default 2:2:2)
  --workdir DIR       where temp frames go (default: system temp)
  --force             process even if the input is already 4K
  --keep-temp         keep temporary frames (debugging)
  --setup-only        just download the tools and exit
  --verbose           print every command that runs
```

## Troubleshooting

- **It's running but VERY slowly, or it warned about Vulkan/GPU** — the AI
  engine silently falls back to your CPU when no usable Vulkan driver is
  found (the tool warns when it detects this). Update your GPU driver:
  NVIDIA (nvidia.com), AMD (Adrenalin), or Intel (intel.com/download-center),
  then run again.
- **Out of VRAM / "upscale incomplete"** — the tool auto-retries with
  smaller tiles, but you can pin it: `--tile 128`.
- **It's using the wrong GPU** (dual-GPU laptop) — try `--gpu 1`.
- **Output is `.mkv` instead of `.mp4`** — your clip's audio codec (e.g.
  Opus/FLAC) can't be stored in MP4 without re-encoding; MKV keeps it
  bit-identical. Every player (VLC, MPC-HC, mpv) handles MKV fine.
- **Input is already 4K** — the tool refuses by default; use `--force`.
- **macOS: "cannot be opened because the developer cannot be verified"** —
  run `xattr -dr com.apple.quarantine bin/realesrgan` once (from this
  project's folder).

## Credits

- AI models & engine: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  (`realesr-animevideov3`) running on [ncnn](https://github.com/Tencent/ncnn)
  Vulkan (BSD-3-Clause).
- Video plumbing: [FFmpeg](https://ffmpeg.org/).
