# ANIME 4K UPSCALER

Paste an anime clip → get an **Ultra 4K UHD** version, upscaled by AI on
**your laptop's GPU**. The audio track is **copied bit-for-bit** — it sounds
exactly the same as the original.

- Works on **any laptop GPU**: NVIDIA, AMD, or Intel (including integrated
  graphics) — it uses Vulkan, so there's no CUDA/PyTorch to install.
- **Zero setup**: one Python file, no pip packages. On first run it
  auto-downloads the AI engine (Real-ESRGAN) and, if needed, ffmpeg.
- **Built for speed**: anime-specific fast model (`realesr-animevideov3`),
  smart scale factor (1080p only needs 2×, not 4×), hardware video encoding
  (NVENC / QuickSync / AMF auto-detected), and the decode → GPU-upscale →
  encode stages run **in parallel**.

## Quick start

**Windows (easiest):** drag your clip onto **`Upscale to 4K.bat`**.
Or double-click it and paste the clip's path when asked.

**Any OS:**

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
clip.mp4 ─┬─ video ─▶ decode frames ─▶ Real-ESRGAN (GPU, Vulkan) ─▶ fit to 3840×2160 ─▶ encode ─┐
          │              (chunked, runs in parallel with the GPU and the encoder)               ├─▶ clip_4K.mp4
          └─ audio ────────────────────────── copied unchanged (bit-for-bit) ──────────────────┘
```

Frames are processed in small chunks so temp disk usage stays bounded, and
the three stages overlap: while the GPU upscales chunk *N*, the CPU is
already decoding chunk *N+1* and encoding chunk *N−1*.

The scale factor is picked automatically — the smallest one that reaches 4K
(1080p → 2×, 720p → 3×, ≤540p → 4×), because upscaling less is *much*
faster and any small remainder is finished with a high-quality Lanczos
resize.

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
python upscale.py INPUT [-o OUTPUT]
  --fast              speed mode (jpeg intermediates + faster encode preset)
  --scale {auto,2,3,4}  AI scale factor (default: auto)
  --model {animevideo,anime-sharp,photo}
                      animevideo = fast anime video model (default)
                      anime-sharp = crisper lines, ~4x slower
  --encoder NAME      force an ffmpeg encoder (default: auto-detect hardware)
  --quality N         CRF/CQ quality, lower = better (default 16–19)
  --chunk N           frames per chunk (default 150; bounds temp disk usage)
  --tile N            GPU tile size (try 256 or 128 if you run out of VRAM)
  --gpu N             GPU index (dual-GPU laptops: try 1)
  --workdir DIR       where temp frames go (default: system temp)
  --setup-only        just download the tools and exit
```

## Troubleshooting

- **"vkCreateInstance failed" / upscaler crashes instantly** — update your
  GPU driver (Vulkan comes with it). NVIDIA: GeForce Experience / nvidia.com;
  AMD: Adrenalin; Intel: intel.com/download-center.
- **Out of VRAM / upscaler dies mid-chunk** — the tool auto-retries with
  smaller tiles, but you can pin it: `--tile 128`.
- **It's using the wrong GPU** (dual-GPU laptop) — try `--gpu 1`.
- **Output is `.mkv` instead of `.mp4`** — your clip's audio codec (e.g.
  Opus/FLAC) can't be stored in MP4 without re-encoding; MKV keeps it
  bit-identical. Every player (VLC, MPC-HC, mpv) handles MKV fine.
- **Input is already 4K** — the tool refuses by default; use `--force`.
- **macOS: "cannot be opened because the developer cannot be verified"** —
  run `xattr -dr com.apple.quarantine bin/realesrgan` once.

## Credits

- AI models & engine: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  (`realesr-animevideov3`) running on [ncnn](https://github.com/Tencent/ncnn) Vulkan.
- Video plumbing: [FFmpeg](https://ffmpeg.org/).
