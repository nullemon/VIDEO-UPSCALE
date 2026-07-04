#!/usr/bin/env python3
"""
ANIME 4K UPSCALER
=================
Paste an anime clip -> get an Ultra 4K UHD version. Audio is copied
bit-for-bit (sound stays identical). Runs on your laptop GPU (NVIDIA,
AMD or Intel) through Vulkan -- no CUDA / PyTorch install needed.

Quick start:
    python upscale.py                 (then paste the path of your clip)
    python upscale.py "my clip.mp4"   (direct)
    python upscale.py clip.mp4 --fast (speed over max quality)

First run auto-downloads the AI engine (Real-ESRGAN ncnn Vulkan) and,
on Windows, ffmpeg. Everything lands in ./bin next to this script.

No pip packages required -- Python 3.8+ standard library only.

Pipeline: the source video is decoded exactly ONCE into a raw frame
stream (seam-proof: chunks are split by frame count, never by seeking),
pumped into per-chunk image dirs, upscaled on the GPU, and encoded --
with decode, GPU work and encoding all running in parallel.
"""

import argparse
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from fractions import Fraction

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TARGET_LONG = 3840   # 4K UHD box (long side)
TARGET_SHORT = 2160  # 4K UHD box (short side)

OS_TAG = {"Windows": "windows", "Linux": "ubuntu", "Darwin": "macos"}

ENGINES = {
    # Real-ESRGAN: the classic; animevideo model is fast, x4plus-anime sharp
    "esrgan": {
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-{os}.zip"),
        "exe": "realesrgan-ncnn-vulkan",
        "marker": "models",           # dir that must sit next to the exe
        "label": "Real-ESRGAN AI upscaler",
    },
    # Real-CUGAN: anime-specific, with built-in compression-artifact removal
    "cugan": {
        "url": ("https://github.com/nihui/realcugan-ncnn-vulkan/releases/"
                "download/20220728/realcugan-ncnn-vulkan-20220728-{os}.zip"),
        "exe": "realcugan-ncnn-vulkan",
        "marker": "models-se",
        "label": "Real-CUGAN AI upscaler",
    },
    # RIFE: motion interpolation (frame doubling) for silky reel-style motion
    "rife": {
        "url": ("https://github.com/nihui/rife-ncnn-vulkan/releases/"
                "download/20221029/rife-ncnn-vulkan-20221029-{os}.zip"),
        "exe": "rife-ncnn-vulkan",
        "marker": "rife-v4.6",
        "label": "RIFE motion interpolator",
    },
}

# The "reel look": deband gradients, crisp lines, punchy color, soft bloom.
EYECANDY_VF = (
    "deband,"
    "cas=0.45,"
    "vibrance=intensity=0.12,"
    "eq=contrast=1.05,"
    "split[ec_m][ec_b];"
    "[ec_b]gblur=sigma=14[ec_g];"
    "[ec_m][ec_g]blend=all_mode=screen:all_opacity=0.15"
)

FFMPEG_WIN_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_LINUX_URL = (
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
)

MODELS = {
    # key: (model name passed to realesrgan, allowed scales, blurb)
    "animevideo": ("realesr-animevideov3", (2, 3, 4), "fast anime-video model (default)"),
    "anime-sharp": ("realesrgan-x4plus-anime", (4,), "sharper lines, ~4x slower"),
    "photo": ("realesrgan-x4plus", (4,), "general/photo model, slow"),
}

# Audio codecs that can be stream-copied into an .mp4 container.
MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "alac"}

# Hardware encoders to probe, best-first. libx264 is the universal fallback.
HW_ENCODERS = ["hevc_nvenc", "h264_nvenc", "hevc_qsv", "h264_qsv",
               "hevc_amf", "h264_amf"]

MAX_PENDING_ENCODES = 2   # encoder processes allowed to run behind the GPU
CHUNK_DIRS_AHEAD = 2      # extracted-but-not-yet-upscaled chunk dirs on disk

VERBOSE = False

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def info(msg):
    print(msg, flush=True)


def warn(msg):
    print(f"[!] {msg}", flush=True)


def die(msg, code=1):
    print(f"\n[X] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def fmt_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


class ProcRegistry:
    """Tracks live child processes so any abort can kill them all."""

    def __init__(self):
        self._procs = set()
        self._lock = threading.Lock()

    def add(self, proc):
        with self._lock:
            self._procs.add(proc)

    def discard(self, proc):
        with self._lock:
            self._procs.discard(proc)

    def kill_all(self):
        """Kill every registered child and wait until each has exited,
        so temp files are no longer held open when cleanup runs."""
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                if os.name != "nt":
                    # kill the whole process group so children of children
                    # (e.g. wrapper scripts) cannot survive and keep writing
                    try:
                        os.killpg(os.getpgid(p.pid), 9)
                    except (OSError, ProcessLookupError):
                        p.kill()
                else:
                    p.kill()
            except OSError:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        with self._lock:
            self._procs.difference_update(procs)


def _popen_kwargs():
    """Children get their own process group on POSIX so kill_all can
    reap grandchildren too."""
    if os.name != "nt":
        return {"start_new_session": True}
    return {}


PROCS = ProcRegistry()


def _print_cmd(cmd):
    if VERBOSE:
        info("  $ " + " ".join(str(c) for c in cmd))


def run(cmd, desc="command", capture=False, want_err=False):
    """Run a subprocess to completion; raise RuntimeError with stderr tail on failure.
    Returns stdout text, or stderr text when want_err=True."""
    _print_cmd(cmd)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **_popen_kwargs(),
    )
    PROCS.add(proc)
    try:
        out, err = proc.communicate()
    except BaseException:
        # interrupted while the child is still alive: leave it registered
        # so kill_all() can terminate it; unregister only if already dead.
        if proc.poll() is not None:
            PROCS.discard(proc)
        raise
    PROCS.discard(proc)
    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-8:]
        raise RuntimeError(f"{desc} failed (exit {proc.returncode}):\n  " + "\n  ".join(tail))
    if want_err:
        return (err or b"").decode("utf-8", "replace")
    return (out or b"").decode("utf-8", "replace")


def start(cmd, desc="command", stdin=None, stdout=None, stderr_path=None):
    """Start a subprocess without waiting.

    stderr goes to a file (never an unread pipe, which could deadlock).
    Returns a handle dict for finish().
    """
    _print_cmd(cmd)
    err_fh = open(stderr_path, "wb") if stderr_path else None
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdin=stdin if stdin is not None else subprocess.DEVNULL,
        stdout=stdout if stdout is not None else subprocess.DEVNULL,
        stderr=err_fh if err_fh else subprocess.DEVNULL,
        **_popen_kwargs(),
    )
    PROCS.add(proc)
    return {"proc": proc, "desc": desc, "err_path": stderr_path, "err_fh": err_fh}


def finish(handle):
    """Wait for a process from start(); raise on failure."""
    proc = handle["proc"]
    try:
        proc.wait()
    except BaseException:
        if proc.poll() is not None:
            PROCS.discard(proc)
        raise
    PROCS.discard(proc)
    if handle["err_fh"]:
        try:
            handle["err_fh"].close()
        except OSError:
            pass
    if proc.returncode != 0:
        tail = ""
        if handle["err_path"] and os.path.exists(handle["err_path"]):
            with open(handle["err_path"], "rb") as f:
                tail = f.read().decode("utf-8", "replace").strip()
            tail = "\n  ".join(tail.splitlines()[-8:])
        raise RuntimeError(f"{handle['desc']} failed (exit {proc.returncode}):\n  {tail}")


# --------------------------------------------------------------------------
# Tool bootstrap (ffmpeg + Real-ESRGAN auto-download)
# --------------------------------------------------------------------------


def download(url, dest, label):
    info(f"[*] Downloading {label} ...")
    info(f"    {url}")
    tmp = dest + ".part"

    def hook(blocks, block_size, total):
        got = blocks * block_size
        if total > 0:
            pct = min(100.0, got * 100.0 / total)
            sys.stdout.write(f"\r    {pct:5.1f}%  ({got // (1 << 20)} MB / {total // (1 << 20)} MB)")
        else:
            sys.stdout.write(f"\r    {got // (1 << 20)} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"download of {label} failed: {e}") from e
    sys.stdout.write("\n")
    os.replace(tmp, dest)


def extract_archive(archive, dest):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.endswith((".tar.xz", ".tar.gz", ".txz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise RuntimeError(f"unknown archive type: {archive}")


def find_in_tree(root, filename):
    for base, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(base, filename)
    return None


def ensure_ffmpeg(bin_dir):
    """Return (ffmpeg, ffprobe) paths; download static builds if missing."""
    exe = ".exe" if os.name == "nt" else ""
    # 1) already on PATH
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ff and fp:
        return ff, fp
    # 2) previously downloaded
    ff = find_in_tree(bin_dir, "ffmpeg" + exe)
    fp = find_in_tree(bin_dir, "ffprobe" + exe)
    if ff and fp:
        return ff, fp
    # 3) download
    os.makedirs(bin_dir, exist_ok=True)
    system = platform.system()
    if system == "Windows":
        archive = os.path.join(bin_dir, "ffmpeg.zip")
        download(FFMPEG_WIN_URL, archive, "ffmpeg (one-time, ~90 MB)")
    elif system == "Linux":
        archive = os.path.join(bin_dir, "ffmpeg.tar.xz")
        download(FFMPEG_LINUX_URL, archive, "ffmpeg static build (one-time, ~40 MB)")
    else:
        die("ffmpeg not found. Install it first, e.g.:  brew install ffmpeg")
    extract_archive(archive, bin_dir)
    os.remove(archive)
    ff = find_in_tree(bin_dir, "ffmpeg" + exe)
    fp = find_in_tree(bin_dir, "ffprobe" + exe)
    if not (ff and fp):
        die("ffmpeg download did not contain expected binaries")
    if os.name != "nt":
        os.chmod(ff, 0o755)
        os.chmod(fp, 0o755)
    return ff, fp


def ensure_upscaler(bin_dir, engine):
    """Return (exe, engine_root_dir) for an AI engine; download if missing."""
    cfg = ENGINES[engine]
    system = platform.system()
    if system not in OS_TAG:
        die(f"unsupported OS for the GPU upscaler: {system}")
    exe_name = cfg["exe"] + (".exe" if os.name == "nt" else "")

    # allow a user-provided binary on PATH (model dir must sit next to it)
    on_path = shutil.which(cfg["exe"])
    if on_path:
        root = os.path.dirname(os.path.abspath(on_path))
        if os.path.isdir(os.path.join(root, cfg["marker"])):
            return on_path, root

    # previously downloaded (search all of bin/ -- exe names are unique,
    # and this also finds pre-rename legacy locations)
    exe = find_in_tree(bin_dir, exe_name)
    if exe:
        root = os.path.dirname(exe)
        if os.path.isdir(os.path.join(root, cfg["marker"])):
            return exe, root

    url = cfg["url"].format(os=OS_TAG[system])
    target = os.path.join(bin_dir, engine)
    os.makedirs(target, exist_ok=True)
    archive = os.path.join(bin_dir, engine + ".zip")
    download(url, archive, f"{cfg['label']} (one-time, ~40 MB)")
    extract_archive(archive, target)
    os.remove(archive)
    exe = find_in_tree(target, exe_name)
    if not exe:
        die(f"{cfg['label']} download did not contain the expected binary")
    if os.name != "nt":
        os.chmod(exe, 0o755)
    root = os.path.dirname(exe)
    if not os.path.isdir(os.path.join(root, cfg["marker"])):
        die(f"{cfg['label']} download did not contain the models folder")
    return exe, root


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


class VideoInfo:
    def __init__(self, width, height, fps, duration, est_frames,
                 audio_codecs, video_codec, color_space, sar):
        self.width = width               # storage dims AFTER rotation
        self.height = height
        self.fps = fps                    # Fraction
        self.duration = duration         # float seconds (0 if unknown)
        self.est_frames = est_frames     # int estimate, for progress only
        self.audio_codecs = audio_codecs  # list of codec names ([] = no audio)
        self.video_codec = video_codec
        self.color_space = color_space   # ffprobe color_space or None
        self.sar = sar                   # sample aspect ratio (Fraction, 1 if square)

    def display_dims(self):
        """Pixel dims as they appear on screen (anamorphic-corrected)."""
        return float(self.width * self.sar), float(self.height)


def probe(ffprobe, path):
    out = run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        desc="ffprobe", capture=True,
    )
    data = json.loads(out)
    vstream = None
    audio_codecs = []
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and vstream is None:
            # skip attached cover art
            if s.get("disposition", {}).get("attached_pic"):
                continue
            vstream = s
        elif s.get("codec_type") == "audio":
            audio_codecs.append(s.get("codec_name", "unknown"))
    if vstream is None:
        die("no video stream found in the input file")

    def parse_rate(txt):
        try:
            frac = Fraction(txt)
            return frac if frac > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    fps = parse_rate(vstream.get("avg_frame_rate", "0/0")) or \
        parse_rate(vstream.get("r_frame_rate", "0/0"))
    if fps is None:
        warn("could not detect frame rate; assuming 24000/1001 (23.976)")
        fps = Fraction(24000, 1001)
    # guard against bogus rates from broken headers
    if fps > 240:
        alt = parse_rate(vstream.get("r_frame_rate", "0/0"))
        fps = alt if alt and alt <= 240 else Fraction(24000, 1001)

    duration = 0.0
    for src in (vstream.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue

    est_frames = 0
    try:
        est_frames = int(vstream.get("nb_frames", 0))
    except (TypeError, ValueError):
        pass
    if est_frames <= 0 and duration > 0:
        est_frames = int(duration * fps) + 1

    # rotation metadata (phone clips): ffmpeg autorotates while decoding,
    # so the rotated dimensions are the real ones for the whole pipeline
    rot = 0
    for sd in vstream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(sd["rotation"]) % 360
            except (TypeError, ValueError):
                pass
    if not rot:
        try:
            rot = int((vstream.get("tags") or {}).get("rotate", 0)) % 360
        except (TypeError, ValueError):
            pass
    width, height = int(vstream["width"]), int(vstream["height"])

    sar = Fraction(1)
    try:
        cand = Fraction(vstream.get("sample_aspect_ratio", "").replace(":", "/"))
        if cand > 0:
            sar = cand
    except (ValueError, ZeroDivisionError):
        pass
    if rot in (90, 270):
        width, height = height, width
        if sar != 1:
            sar = 1 / sar

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        est_frames=max(est_frames, 1),
        audio_codecs=audio_codecs,
        video_codec=vstream.get("codec_name", "?"),
        color_space=vstream.get("color_space"),
        sar=sar,
    )


def source_color_matrix(vinfo):
    """Colorimetry of the source, for correct YUV<->RGB conversion.
    Falls back to the standard guess by resolution when untagged."""
    tags = {
        "bt709": "bt709",
        "bt470bg": "bt601",
        "smpte170m": "bt601",
        "bt2020nc": "bt2020",
        "bt2020c": "bt2020",
    }
    mat = tags.get(vinfo.color_space or "")
    if mat:
        return mat
    return "bt709" if min(vinfo.width, vinfo.height) >= 720 else "bt601"


def ffmpeg_supports_fps_mode(ffmpeg):
    """-fps_mode replaced -vsync in ffmpeg 5.1."""
    try:
        out = run([ffmpeg, "-version"], desc="ffmpeg -version", capture=True)
        m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2))) >= (5, 1)
        m = re.search(r"ffmpeg version n?(\d+)", out)
        if m:
            return int(m.group(1)) >= 6
    except RuntimeError:
        pass
    return True  # assume modern (git/dev builds)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def choose_ai_scale(width, height, allowed):
    """Smallest allowed AI factor that reaches the 4K box (fit-inside)."""
    long_side, short_side = max(width, height), min(width, height)
    fit = min(TARGET_LONG / long_side, TARGET_SHORT / short_side)
    for s in sorted(allowed):
        if s >= fit:
            return s, fit
    return max(allowed), fit


def target_box(width, height):
    """4K UHD box, oriented to match the source."""
    if height > width:
        return TARGET_SHORT, TARGET_LONG
    return TARGET_LONG, TARGET_SHORT


def final_dims(disp_w, disp_h):
    """Final output size from DISPLAY dims (anamorphic-corrected):
    fit inside the 4K box, keep aspect, force even."""
    tw, th = target_box(disp_w, disp_h)
    ratio = min(tw / disp_w, th / disp_h)
    w = int(disp_w * ratio + 0.5) // 2 * 2
    h = int(disp_h * ratio + 0.5) // 2 * 2
    return max(w, 2), max(h, 2)


def scale_filter(out_w, out_h, in_matrix=None):
    inm = f"in_color_matrix={in_matrix}:" if in_matrix else ""
    return (f"scale=w={out_w}:h={out_h}:flags=lanczos:"
            f"{inm}out_color_matrix=bt709,setsar=1")


def detect_encoder(ffmpeg, forced=None):
    """Pick the fastest working encoder by actually test-encoding frames."""
    candidates = [forced] if forced else HW_ENCODERS + ["libx264"]
    for enc in candidates:
        cmd = [ffmpeg, "-v", "error", "-f", "lavfi",
               "-i", "color=black:s=256x256:d=0.2:r=24",
               "-frames:v", "3", "-c:v", enc]
        cmd += encoder_quality_args(enc, quality=None, fast=True)
        cmd += ["-f", "null", "-"]
        try:
            run(cmd, desc=f"encoder test ({enc})")
            return enc
        except RuntimeError:
            if forced:
                die(f"requested encoder '{forced}' is not usable on this machine")
            continue
    die("no working video encoder found in ffmpeg")


def encoder_quality_args(enc, quality, fast):
    """Quality/preset flags per encoder family. quality=None -> default."""
    args = []
    if enc.endswith("_nvenc"):
        q = quality if quality is not None else 19
        args += ["-preset", "p4" if fast else "p5", "-tune", "hq",
                 "-rc", "vbr", "-cq", str(q), "-b:v", "0"]
    elif enc.endswith("_qsv"):
        q = quality if quality is not None else 19
        args += ["-global_quality", str(q), "-preset",
                 "fast" if fast else "slow"]
    elif enc.endswith("_amf"):
        q = quality if quality is not None else 19
        args += ["-quality", "speed" if fast else "quality",
                 "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q)]
    else:  # libx264
        q = quality if quality is not None else (18 if fast else 16)
        args += ["-crf", str(q), "-preset", "veryfast" if fast else "medium"]
    if enc.startswith("hevc"):
        args += ["-tag:v", "hvc1"]
    return args


# --------------------------------------------------------------------------
# Progress display
# --------------------------------------------------------------------------


class Progress:
    """Single-line progress bar: frames upscaled / total, rate, ETA."""

    def __init__(self, total_frames):
        self.total = max(total_frames, 1)
        self.done_base = 0          # frames from completed chunks
        self.current_dir = None     # out-dir of the chunk being upscaled
        self.lock = threading.Lock()
        self.start_time = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def set_current(self, out_dir):
        with self.lock:
            self.current_dir = out_dir

    def chunk_done(self, frames):
        with self.lock:
            self.done_base += frames
            self.current_dir = None

    def _count(self):
        with self.lock:
            base, cur = self.done_base, self.current_dir
        extra = 0
        if cur and os.path.isdir(cur):
            try:
                extra = len(os.listdir(cur))
            except OSError:
                extra = 0
        return base + extra

    def _loop(self):
        while not self._stop.wait(0.5):
            self._render()

    def _render(self, final=False):
        done = min(self._count(), self.total)
        elapsed = time.time() - self.start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - done) / rate if rate > 0 else 0
        pct = done * 100.0 / self.total
        if not final:
            pct = min(pct, 99.9)
        width = 26
        filled = int(width * pct / 100)
        bar = "#" * filled + "-" * (width - filled)
        eta = fmt_time(remaining) if (rate > 0 and not final) else "--:--"
        if final:
            eta = fmt_time(0)
        line = (f"\r  [{bar}] {pct:5.1f}%  {done}/{self.total} frames"
                f"  {rate:5.1f} f/s  ETA {eta}   ")
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        self._stop.set()
        self._thread.join(timeout=2)
        with self.lock:
            self.done_base = self.total
            self.current_dir = None
        self._render(final=True)
        sys.stdout.write("\n")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def read_exact(stream, n):
    """Read exactly n bytes; None on clean EOF; raise on truncated frame."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = stream.readinto(view[got:])
        if not r:
            if got == 0:
                return None
            raise RuntimeError(
                f"video decoder stopped mid-frame ({got}/{n} bytes) -- "
                f"the input file may be corrupt")
        got += r
    return bytes(buf)


class Pipeline:
    def __init__(self, args, tools, vinfo):
        self.args = args
        self.ffmpeg = tools["ffmpeg"]
        self.upscaler = tools["upscaler"]
        self.engine_root = tools["engine_root"]
        self.v = vinfo
        if args.engine == "cugan":
            self.model_name = None
            self.denoise = args.denoise
            allowed = (2, 3, 4)
        elif args.custom_model:
            self.model_name = args.custom_model
            self.denoise = None
            allowed = (2, 3, 4)
        else:
            self.model_name, allowed, _ = MODELS[args.model]
            self.denoise = None
        disp_w, disp_h = vinfo.display_dims()
        if args.scale != "auto":
            s = int(args.scale)
            if s not in allowed:
                die(f"model '{args.model}' only supports scales {allowed}")
            self.ai_scale = s
        else:
            self.ai_scale, _fit = choose_ai_scale(disp_w, disp_h, allowed)
        if args.engine == "cugan" and self.denoise in (1, 2) \
                and self.ai_scale != 2:
            warn(f"denoise {self.denoise} only exists for 2x; using 3")
            self.denoise = 3
        self.img_ext = "jpg" if args.fast else "png"
        self.fps_str = f"{vinfo.fps.numerator}/{vinfo.fps.denominator}"
        self.use_fps_mode = ffmpeg_supports_fps_mode(self.ffmpeg)
        self.encoder = detect_encoder(self.ffmpeg, args.encoder)
        self.out_w, self.out_h = final_dims(disp_w, disp_h)
        self.src_matrix = source_color_matrix(vinfo)
        self.frame_bytes = vinfo.width * vinfo.height * 3  # rgb24
        self._cpu_warned = False
        f2 = vinfo.fps * 2
        self.fps2_str = f"{f2.numerator}/{f2.denominator}"
        self.rife = tools.get("rife")
        self.rife_root = tools.get("rife_root")
        self.preview = None
        if args.preview:
            dur = 8.0
            ss = max(0.0, vinfo.duration / 2 - dur / 2) if vinfo.duration else 0.0
            self.preview = (ss, dur)
            vinfo.est_frames = min(vinfo.est_frames, int(dur * vinfo.fps) + 2)

    # ---- commands ------------------------------------------------------

    def decoder_cmd(self):
        """ONE pass over the source: decode -> CFR -> rgb24 raw stream.
        No seeking anywhere, so chunk seams can never drop/dup frames."""
        cfr = ["-fps_mode", "cfr"] if self.use_fps_mode else ["-vsync", "cfr"]
        seek = ["-ss", f"{self.preview[0]:.3f}"] if self.preview else []
        limit = ["-t", f"{self.preview[1]:.3f}"] if self.preview else []
        return [self.ffmpeg, "-v", "error", "-nostdin"] + seek + \
               ["-i", self.args.input,
                "-map", "0:v:0",
                "-vf", (f"scale=in_color_matrix={self.src_matrix}:"
                        f"flags=lanczos+full_chroma_int+accurate_rnd"),
                "-r", self.fps_str] + cfr + limit + \
               ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    def writer_cmd(self, out_dir):
        """Raw rgb24 frames from stdin -> numbered images in out_dir."""
        cmd = [self.ffmpeg, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.v.width}x{self.v.height}",
               "-framerate", self.fps_str, "-i", "-"]
        if self.img_ext == "jpg":
            # rgb->yuv here uses swscale's bt601 default, which is exactly
            # what JFIF decoders (incl. the upscaler) assume -- consistent.
            cmd += ["-q:v", "2", "-pix_fmt", "yuvj444p"]
        else:
            cmd += ["-compression_level", "1"]
        cmd += [os.path.join(out_dir, f"%08d.{self.img_ext}")]
        return cmd

    def _jobs_and_gpu(self):
        jobs = self.args.jobs
        gpu = self.args.gpu
        if gpu and "," in gpu:
            # multi-GPU: ncnn wants one proc-thread count per GPU
            parts = jobs.split(":")
            if len(parts) == 3 and "," not in parts[1]:
                parts[1] = ",".join([parts[1]] * len(gpu.split(",")))
                jobs = ":".join(parts)
        return jobs, gpu

    def upscale_cmd(self, in_dir, out_dir, tile):
        jobs, gpu = self._jobs_and_gpu()
        if self.args.engine == "cugan":
            # the "pro" models are higher quality but only exist for
            # 2x/3x with denoise -1/0/3
            mdir = os.path.join(self.engine_root, "models-pro")
            if not (self.ai_scale in (2, 3) and self.denoise in (-1, 0, 3)
                    and os.path.isdir(mdir)):
                mdir = os.path.join(self.engine_root, "models-se")
            cmd = [self.upscaler, "-i", in_dir, "-o", out_dir,
                   "-n", str(self.denoise), "-s", str(self.ai_scale),
                   "-m", mdir, "-f", self.img_ext, "-j", jobs]
        else:
            cmd = [self.upscaler, "-i", in_dir, "-o", out_dir,
                   "-n", self.model_name, "-s", str(self.ai_scale),
                   "-m", os.path.join(self.engine_root, "models"),
                   "-f", self.img_ext, "-j", jobs]
        if tile:
            cmd += ["-t", str(tile)]
        if gpu is not None:
            cmd += ["-g", gpu]
        return cmd

    def rife_cmd(self, in_dir, out_dir, target):
        jobs, gpu = self._jobs_and_gpu()
        cmd = [self.rife, "-i", in_dir, "-o", out_dir,
               "-m", os.path.join(self.rife_root, "rife-v4.6"),
               "-n", str(target),
               "-f", f"%08d.{self.img_ext}", "-j", jobs]
        if self.out_w * self.out_h >= 3200 * 1800:
            cmd += ["-u"]  # UHD mode
        if gpu is not None:
            cmd += ["-g", gpu]
        return cmd

    def encode_cmd(self, in_dir, out_file, fps=None):
        # png intermediates are RGB (matrix applies on rgb->yuv output only);
        # jpg intermediates are YUV that JFIF convention codes as bt601.
        in_matrix = "bt601" if self.img_ext == "jpg" else None
        vf = scale_filter(self.out_w, self.out_h, in_matrix)
        if self.args.eyecandy:
            vf += "," + EYECANDY_VF
        cmd = [self.ffmpeg, "-v", "error", "-y",
               "-framerate", fps or self.fps_str,
               "-i", os.path.join(in_dir, f"%08d.{self.img_ext}"),
               "-vf", vf,
               "-c:v", self.encoder]
        cmd += encoder_quality_args(self.encoder, self.args.quality, self.args.fast)
        cmd += ["-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709",
                out_file]
        return cmd

    # ---- pump thread: raw stream -> per-chunk image dirs -----------------

    def _pump(self, dec_handle, workdir, out_q, sem, stop):
        """Slices the decoder's raw frame stream into chunk dirs.
        Backpressure: waits on `sem` before starting a new chunk dir, which
        blocks the decoder on its stdout pipe -- disk usage stays bounded."""
        dec = dec_handle["proc"]
        try:
            idx = 0
            eof = False
            while not eof:
                # backpressure gate
                while not sem.acquire(timeout=0.5):
                    if stop.is_set():
                        return
                if stop.is_set():
                    return
                in_dir = os.path.join(workdir, f"in_{idx:05d}")
                os.makedirs(in_dir, exist_ok=True)
                writer = start(
                    self.writer_cmd(in_dir),
                    desc=f"frame writer (chunk {idx})",
                    stdin=subprocess.PIPE,
                    stderr_path=os.path.join(workdir, f"writer_{idx:05d}.log"),
                )
                n = 0
                try:
                    while n < self.args.chunk and not stop.is_set():
                        frame = read_exact(dec.stdout, self.frame_bytes)
                        if frame is None:
                            eof = True
                            break
                        writer["proc"].stdin.write(frame)
                        n += 1
                finally:
                    try:
                        writer["proc"].stdin.close()
                    except OSError:
                        pass
                finish(writer)
                if stop.is_set():
                    return
                if n > 0:
                    out_q.put(("chunk", idx, in_dir, n))
                    idx += 1
                else:
                    shutil.rmtree(in_dir, ignore_errors=True)
                    sem.release()
            # propagate decoder failure (e.g. corrupt input) as an error
            finish(dec_handle)
            out_q.put(("done", None, None, None))
        except BaseException as e:  # noqa: BLE001 - forwarded to main thread
            out_q.put(("error", e, None, None))

    # ---- GPU upscale -----------------------------------------------------

    def _warn_cpu_fallback(self, err_text):
        """The upscaler binary silently falls back to CPU when Vulkan is
        unusable -- surface that, or the user just sees 'slow'."""
        if self._cpu_warned or not err_text:
            return
        low = err_text.lower()
        markers = ("vkcreateinstance failed", "no vulkan device",
                   "vkenumeratephysicaldevices failed", "invalid gpu device")
        if any(m in low for m in markers):
            self._cpu_warned = True
            warn("no usable GPU/Vulkan driver found -- the AI upscaler is "
                 "running on your CPU (MUCH slower). Update your GPU driver "
                 "to fix this.")

    def _count_frames(self, d):
        try:
            return len([f for f in os.listdir(d)
                        if f.endswith("." + self.img_ext)])
        except OSError:
            return 0

    def upscale(self, in_dir, out_dir):
        """Blocking GPU upscale, verified by OUTPUT FRAME COUNT.
        The upscaler binary can exit 0 even when frames fail (VRAM/driver
        errors just skip frames), so exit codes alone cannot be trusted --
        missing frames would silently truncate the chunk. Retries with
        smaller GPU tiles, which fixes out-of-VRAM failures."""
        os.makedirs(out_dir, exist_ok=True)
        expected = self._count_frames(in_dir)
        tiles = [self.args.tile] if self.args.tile else [0, 256, 128]
        last_err = None
        for i, tile in enumerate(tiles):
            err_text = ""
            exit_err = None
            try:
                err_text = run(self.upscale_cmd(in_dir, out_dir, tile),
                               desc="AI upscale", want_err=True)
            except RuntimeError as e:
                exit_err = e
            self._warn_cpu_fallback(err_text)
            got = self._count_frames(out_dir)
            if got >= expected and exit_err is None:
                return
            last_err = exit_err or RuntimeError(
                f"AI upscaler produced only {got}/{expected} frames "
                f"(likely out of GPU memory). Try --tile 128 or --fast.")
            if i + 1 < len(tiles):
                warn(f"upscale incomplete ({got}/{expected} frames); "
                     f"retrying with tile size {tiles[i + 1]} ...")
        raise last_err

    def smooth_chunk(self, idx, frames_dir, n, sentinel_src, workdir):
        """Double the frame rate of one chunk with RIFE, seam-correct.

        For every chunk except the last, the FIRST upscaled frame of the
        NEXT chunk is appended as a sentinel so the interpolated frame
        that belongs exactly on the chunk seam gets generated; the
        sentinel itself is dropped afterwards (the next chunk starts
        with it). Output frame counts: 2n per chunk, 2n-1 for the last
        -- identical to interpolating the whole video in one pass."""
        ext = self.img_ext
        if sentinel_src:
            shutil.copyfile(sentinel_src,
                            os.path.join(frames_dir, f"{n + 1:08d}.{ext}"))
            target = 2 * n + 1
        else:
            if n < 2:  # single trailing frame: just duplicate it
                shutil.copyfile(
                    os.path.join(frames_dir, f"{1:08d}.{ext}"),
                    os.path.join(frames_dir, f"{2:08d}.{ext}"))
                return frames_dir, 2
            target = 2 * n - 1
        smooth_dir = os.path.join(workdir, f"smooth_{idx:05d}")
        os.makedirs(smooth_dir, exist_ok=True)
        run(self.rife_cmd(frames_dir, smooth_dir, target),
            desc="motion smoothing (RIFE)")
        got = self._count_frames(smooth_dir)
        if got < target:
            raise RuntimeError(
                f"motion smoothing produced {got}/{target} frames -- "
                f"retry without --smooth (or with --tile 128)")
        kept = target
        if sentinel_src:
            # drop the sentinel endpoint; the next chunk begins with it
            os.remove(os.path.join(smooth_dir, f"{target:08d}.{ext}"))
            kept = target - 1
        if not self.args.keep_temp:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return smooth_dir, kept

    # ---- main loop -------------------------------------------------------

    def process(self, workdir):
        """decode(1 pass) -> chunk dirs -> GPU upscale -> parallel encodes.
        Returns the list of encoded chunk files, in order."""
        chunk_files = []
        pending = []          # [(handle, dirs_to_delete_after)]
        out_q = queue.Queue()
        sem = threading.Semaphore(CHUNK_DIRS_AHEAD)
        stop = threading.Event()
        progress = Progress(self.v.est_frames)
        progress.start()

        dec = start(self.decoder_cmd(), desc="video decode",
                    stdout=subprocess.PIPE,
                    stderr_path=os.path.join(workdir, "decoder.log"))
        pump = threading.Thread(target=self._pump,
                                args=(dec, workdir, out_q, sem, stop),
                                daemon=True)
        pump.start()

        def reap(block_all=False):
            while pending and (block_all or len(pending) >= MAX_PENDING_ENCODES
                               or pending[0][0]["proc"].poll() is not None):
                handle, dirs = pending.pop(0)
                finish(handle)
                if not self.args.keep_temp:
                    for d in dirs:
                        shutil.rmtree(d, ignore_errors=True)

        def dispatch(idx, frames_dir, fps_str):
            chunk_file = os.path.join(workdir, f"chunk_{idx:05d}.mp4")
            enc = start(self.encode_cmd(frames_dir, chunk_file, fps_str),
                        desc=f"encode (chunk {idx})",
                        stderr_path=os.path.join(workdir, f"encode_{idx:05d}.log"))
            pending.append((enc, [frames_dir]))
            chunk_files.append(chunk_file)
            reap()

        held = None  # (idx, out_dir, n): chunk awaiting its seam frame

        try:
            while True:
                kind, a, b, c = out_q.get()
                if kind == "error":
                    raise a
                if kind == "done":
                    break
                idx, in_dir, n = a, b, c

                out_dir = os.path.join(workdir, f"out_{idx:05d}")
                progress.set_current(out_dir)
                self.upscale(in_dir, out_dir)
                progress.chunk_done(n)
                # input frames are no longer needed; free the disk and let
                # the pump start filling the next chunk dir
                if not self.args.keep_temp:
                    shutil.rmtree(in_dir, ignore_errors=True)
                sem.release()

                if not self.args.smooth:
                    dispatch(idx, out_dir, self.fps_str)
                    continue
                # smooth mode: a chunk is finalized only after the next
                # chunk's first upscaled frame exists (seam interpolation)
                if held is not None:
                    hidx, hdir, hn = held
                    seam = os.path.join(out_dir, f"{1:08d}.{self.img_ext}")
                    sdir, _kept = self.smooth_chunk(hidx, hdir, hn, seam, workdir)
                    dispatch(hidx, sdir, self.fps2_str)
                held = (idx, out_dir, n)
            if held is not None:
                hidx, hdir, hn = held
                sdir, _kept = self.smooth_chunk(hidx, hdir, hn, None, workdir)
                dispatch(hidx, sdir, self.fps2_str)
            reap(block_all=True)
            progress.finish()
        except BaseException:
            stop.set()
            progress.stop()
            sys.stdout.write("\n")
            raise
        finally:
            stop.set()
            pump.join(timeout=5)
        return chunk_files

    def mux(self, chunk_files, workdir, out_file):
        """Concat chunks and stream-copy the original audio (bit-identical)."""
        list_file = os.path.join(workdir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for c in chunk_files:
                escaped = c.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        audio_src = ["-i", self.args.input]
        if self.preview:
            audio_src = ["-ss", f"{self.preview[0]:.3f}",
                         "-t", f"{self.preview[1]:.3f}"] + audio_src
        cmd = [self.ffmpeg, "-v", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", list_file] + audio_src + \
              ["-map", "0:v:0", "-map", "1:a?",
               "-c:v", "copy", "-c:a", "copy",
               "-map_metadata", "1"]
        if out_file.lower().endswith(".mp4"):
            cmd += ["-movflags", "+faststart"]
        cmd += [out_file]
        run(cmd, desc="final mux")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def needs_mkv(audio_codecs):
    """True if ANY audio track cannot be stream-copied into .mp4."""
    return any(c not in MP4_AUDIO_OK for c in audio_codecs)


def default_output(input_path, audio_codecs):
    base, _ext = os.path.splitext(input_path)
    ext = ".mkv" if needs_mkv(audio_codecs) else ".mp4"
    return base + "_4K" + ext


def check_output_container(out_file, audio_codecs):
    """Validate the output container BEFORE the expensive upscale, so a bad
    -o extension can't blow up at the final mux hours later."""
    base, ext = os.path.splitext(out_file)
    ext = ext.lower()
    if ext not in (".mp4", ".mov", ".m4v", ".mkv"):
        fixed = base + ".mkv"
        warn(f"container '{ext or '(none)'}' cannot hold the upscaled "
             f"stream copy; writing {os.path.basename(fixed)} instead")
        return check_output_container(fixed, audio_codecs)
    if needs_mkv(audio_codecs) and ext in (".mp4", ".mov", ".m4v"):
        fixed = base + ".mkv"
        bad = [c for c in audio_codecs if c not in MP4_AUDIO_OK]
        warn(f"audio codec '{bad[0]}' cannot be copied into {ext}; "
             f"writing {os.path.basename(fixed)} instead (audio unchanged)")
        return fixed
    return out_file


def interactive_input():
    info("=" * 60)
    info("  ANIME 4K UPSCALER  --  paste your clip's path below")
    info("=" * 60)
    while True:
        try:
            raw = input("\nClip path: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        # Windows "Copy as path" wraps in quotes; drag&drop may too
        path = raw.strip('"').strip("'")
        if path and os.path.isfile(path):
            return path
        # macOS/Linux terminal drag&drop escapes spaces with backslashes
        if path and os.name != "nt":
            unescaped = re.sub(r"\\(.)", r"\1", path)
            if os.path.isfile(unescaped):
                return unescaped
        info("  That file does not exist -- try again (tip: drag the file")
        info("  into this window, or right-click it -> 'Copy as path').")


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="upscale.py",
        description="Upscale an anime clip to Ultra 4K UHD on your laptop GPU. "
                    "Audio is copied unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="video file (omit to be prompted)")
    p.add_argument("-o", "--output", help="output file (default: <name>_4K.mp4)")
    p.add_argument("--fast", action="store_true",
                   help="speed mode: jpeg intermediates + faster encoder preset")
    p.add_argument("--turbo", action="store_true",
                   help="run the machine at full tilt: more concurrent GPU "
                        "jobs (8:4:8), bigger chunks (300). Combine with "
                        "--fast for maximum speed")
    p.add_argument("--best", action="store_true",
                   help="maximum quality: the sharper anime model "
                        "(x4plus-anime, ~3-4x slower), 4x supersampling, "
                        "higher encode quality. Best for low-res or "
                        "heavily compressed sources")
    p.add_argument("--reel", action="store_true",
                   help="the Instagram-reel eye-candy preset: cugan engine "
                        "with strong artifact removal + --eyecandy grade + "
                        "--smooth motion, high encode quality")
    p.add_argument("--smooth", action="store_true",
                   help="double the frame rate with AI motion interpolation "
                        "(RIFE) for silky reel-style motion")
    p.add_argument("--eyecandy", action="store_true",
                   help="reel-style grade: deband + crisp line sharpening + "
                        "vibrance + soft glow bloom")
    p.add_argument("--custom-model", metavar="NAME",
                   help="use a community ncnn model you dropped into the "
                        "esrgan models folder (requires explicit --scale)")
    p.add_argument("--preview", action="store_true",
                   help="render only ~8 seconds from the middle of the clip "
                        "-- fast way to compare settings before a full run")
    p.add_argument("--scale", choices=["auto", "2", "3", "4"], default="auto",
                   help="AI scale factor (auto = smallest that reaches 4K)")
    p.add_argument("--engine", choices=["esrgan", "cugan"],
                   help="AI engine: esrgan=Real-ESRGAN (default); "
                        "cugan=Real-CUGAN, best for compressed/low-quality "
                        "sources (combine with --denoise 3)")
    p.add_argument("--denoise", type=int, choices=[-1, 0, 1, 2, 3], default=0,
                   help="cugan only: compression-artifact removal strength "
                        "(3 = strongest, ideal for social-media rips; "
                        "-1 = conservative)")
    p.add_argument("--model", choices=sorted(MODELS),
                   help="esrgan model (default: animevideo, or anime-sharp "
                        "with --best): "
                        + "; ".join(f"{k}={v[2]}" for k, v in MODELS.items()))
    p.add_argument("--encoder", help="force a specific ffmpeg video encoder "
                                     "(default: auto-detect NVENC/QSV/AMF, else x264)")
    p.add_argument("--quality", type=int,
                   help="encoder quality (CRF/CQ, lower=better; default 16-19)")
    p.add_argument("--chunk", type=int,
                   help="frames per chunk (default 150, turbo 300; bounds "
                        "temp disk usage)")
    p.add_argument("--tile", type=int,
                   help="GPU tile size (default auto; use 256/128 on low VRAM)")
    p.add_argument("--gpu", help="GPU index, or comma list to use several "
                                 "at once (e.g. --gpu 0,1)")
    p.add_argument("--jobs", help="upscaler threads as load:proc:save "
                                  "(default 4:2:4, turbo 8:4:8)")
    p.add_argument("--workdir", help="directory for temp files "
                                     "(default: system temp)")
    p.add_argument("--keep-temp", action="store_true",
                   help="keep temporary frames (debugging)")
    p.add_argument("--force", action="store_true",
                   help="process even if the input is already 4K")
    p.add_argument("--setup-only", action="store_true",
                   help="download the tools and exit")
    p.add_argument("--verbose", action="store_true", help="print every command")
    return p.parse_args(argv)


def cleanup_workdir(workdir):
    """Remove the temp tree; children are already dead (kill_all ran).
    Retry for slow file-handle release (Windows), warn if litter remains."""
    for _attempt in range(4):
        shutil.rmtree(workdir, ignore_errors=True)
        if not os.path.exists(workdir):
            return
        time.sleep(1.0)
    if os.path.exists(workdir):
        warn(f"could not fully remove temp files in {workdir} -- "
             f"you may delete that folder manually")


def main(argv=None):
    global VERBOSE
    args = parse_args(argv)
    VERBOSE = args.verbose
    if args.best and args.fast:
        die("--best and --fast pull in opposite directions -- pick one")
    if args.reel:
        args.eyecandy = True
        args.smooth = True
        if args.quality is None:
            args.quality = 15
        if args.engine is None:
            # default reel path: the SHARP model. Strong cugan denoise
            # melts detail on decent sources -- only use it when asked
            # (--reel --engine cugan --denoise 3 for junky rips).
            args.engine = "esrgan"
            if args.model is None and not args.fast:
                args.best = True
        elif args.engine == "cugan" and args.denoise == 0:
            args.denoise = 3
    if args.engine is None:
        args.engine = "esrgan"
    if args.model is not None and args.engine == "cugan":
        warn("--model applies to the esrgan engine only; ignoring it")
    if args.custom_model:
        if args.engine != "esrgan":
            die("--custom-model works with the esrgan engine only")
        if args.scale == "auto":
            die("--custom-model needs an explicit --scale matching the model")
    if args.model is None:
        args.model = "anime-sharp" if args.best else "animevideo"
    if args.best and args.quality is None:
        args.quality = 15
    if args.best and args.engine == "cugan" and args.denoise == 0:
        args.denoise = 3  # --best on cugan implies strong artifact removal
    if args.jobs is None:
        args.jobs = "8:4:8" if args.turbo else "4:2:4"
    if args.chunk is None:
        # 4x intermediates in --best mode are huge; smaller chunks keep
        # temp disk bounded
        args.chunk = 100 if args.best else (300 if args.turbo else 150)
    if args.chunk < 1:
        die("--chunk must be at least 1")
    if args.quality is not None and not 0 <= args.quality <= 51:
        die("--quality must be between 0 and 51")
    if args.gpu is not None and not re.fullmatch(r"\d+(,\d+)*", args.gpu):
        die("--gpu must be a GPU index or comma list, e.g. 1 or 0,1")

    bin_dir = os.path.join(script_dir(), "bin")
    try:
        ffmpeg, ffprobe = ensure_ffmpeg(bin_dir)
        if args.setup_only:
            info("[+] Tools ready:")
            info(f"    ffmpeg     : {ffmpeg}")
            for eng in ENGINES:
                exe, _root = ensure_upscaler(bin_dir, eng)
                info(f"    {eng:<10} : {exe}")
            return 0
        upscaler, engine_root = ensure_upscaler(bin_dir, args.engine)
        rife = rife_root = None
        if args.smooth:
            rife, rife_root = ensure_upscaler(bin_dir, "rife")
    except RuntimeError as e:
        die(str(e))

    if not args.input:
        args.input = interactive_input()
    args.input = os.path.abspath(args.input.strip('"').strip("'"))
    if not os.path.isfile(args.input):
        die(f"input file not found: {args.input}")

    v = probe(ffprobe, args.input)

    dw, dh = v.display_dims()
    if min(TARGET_LONG / max(dw, dh),
           TARGET_SHORT / min(dw, dh)) <= 1.0 and not args.force:
        die(f"input is already {v.width}x{v.height} (>= 4K). "
            f"Use --force to process anyway.")

    tools = {"ffmpeg": ffmpeg, "ffprobe": ffprobe,
             "upscaler": upscaler, "engine_root": engine_root,
             "rife": rife, "rife_root": rife_root}
    pipe = Pipeline(args, tools, v)
    ow, oh = pipe.out_w, pipe.out_h

    out_file = args.output or default_output(args.input, v.audio_codecs)
    out_file = check_output_container(os.path.abspath(out_file), v.audio_codecs)
    if args.preview:
        base, ext = os.path.splitext(out_file)
        out_file = base + "_preview" + ext

    fps_f = float(v.fps)
    fps_desc = f"{fps_f:.3f} fps"
    if args.smooth:
        fps_desc += f" -> {fps_f * 2:.3f} fps (AI-smoothed)"
    audio_desc = ", ".join(v.audio_codecs) + " (copied bit-for-bit)" \
        if v.audio_codecs else "none"
    info("")
    info(f"  Input   : {os.path.basename(args.input)}")
    info(f"            {v.width}x{v.height} {v.video_codec} @ {fps_desc}"
         f"  |  {fmt_time(v.duration)}  |  ~{v.est_frames} frames")
    info(f"  Audio   : {audio_desc}")
    info(f"  Output  : {os.path.basename(out_file)}  ->  {ow}x{oh} (Ultra 4K UHD)")
    if args.engine == "cugan":
        info(f"  AI      : Real-CUGAN x{pipe.ai_scale} denoise={pipe.denoise}"
             f" on GPU (Vulkan)")
    else:
        info(f"  AI      : {pipe.model_name} x{pipe.ai_scale} on GPU (Vulkan)")
    info(f"  Encoder : {pipe.encoder}"
         + ("  [hardware]" if pipe.encoder != "libx264" else "  [cpu]")
         + ("  |  REEL" if args.reel else "")
         + ("  |  TURBO" if args.turbo else "")
         + ("  |  BEST quality" if args.best else "")
         + ("  |  EYE CANDY" if args.eyecandy and not args.reel else "")
         + ("  |  SMOOTH 2x" if args.smooth and not args.reel else "")
         + ("  |  FAST mode" if args.fast else "")
         + ("  |  PREVIEW ~8s" if args.preview else ""))
    if not args.best and args.engine != "cugan" \
            and min(v.width, v.height) < 700:
        info("  Tip     : low-res source detected -- try --best (sharper "
             "lines) or --engine cugan --denoise 3 (cleans compression)")
    info("")

    workdir_root = args.workdir or tempfile.gettempdir()
    os.makedirs(workdir_root, exist_ok=True)
    free_gb = shutil.disk_usage(workdir_root).free / (1 << 30)
    if free_gb < 8:
        warn(f"only {free_gb:.1f} GB free in {workdir_root}; consider "
             f"--workdir on a bigger drive or --fast (uses ~10x less space)")

    workdir = tempfile.mkdtemp(prefix="anime4k_", dir=workdir_root)
    t0 = time.time()
    try:
        chunks = pipe.process(workdir)
        if not chunks:
            die("no frames could be extracted from the input")
        info("[*] Stitching chunks" +
             (" + copying original audio ..." if v.audio_codecs else " ..."))
        pipe.mux(chunks, workdir, out_file)
    except KeyboardInterrupt:
        die("interrupted", code=130)
    except RuntimeError as e:
        die(str(e))
    except Exception as e:  # unexpected: fail with a readable message
        if VERBOSE:
            import traceback
            traceback.print_exc()
        die(f"unexpected error: {e.__class__.__name__}: {e}")
    finally:
        # kill children FIRST so nothing holds the temp files open,
        # then remove the temp tree
        PROCS.kill_all()
        if not args.keep_temp:
            cleanup_workdir(workdir)

    dt = time.time() - t0
    size_mb = os.path.getsize(out_file) / (1 << 20)
    info("")
    info(f"[+] DONE in {fmt_time(dt)}  ({v.est_frames / dt:.1f} frames/s overall)")
    info(f"[+] {out_file}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
