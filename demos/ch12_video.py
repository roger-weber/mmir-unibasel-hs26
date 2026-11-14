"""Helpers for the chapter 12 demos (Video Features): the film, decoding, caches, and the learned models.

Each demo computes its concept step by step in the notebook (histograms, distances, thresholds,
differencing, Lucas-Kanade, pooling, fusion). This module holds the plumbing that the demos share:

    the film           ensure_film, FPS, N_FRAMES, WIDTH, HEIGHT, CREDIT          (all demos)
    decoding           decode, frames, frame, keyframe_indices                     (all demos)
    caching            cached                                                      (all demos)
    time codes         tc, frame_of                                                (all demos)
    annotation         load_annotation, evaluate                                   (ch12-01)
    TransNet V2        transnet_probabilities, transnet_transitions, shots_from_transitions  (all demos)
    fight sequences    FIGHTS, is_fight                                            (ch12-02, ch12-03)
    motion             film_motion, shot_activity                                  (ch12-02, ch12-03)
    RAFT               raft_flow                                                   (ch12-02)
    CLIP               Clip, clip_frame_embeddings                                 (ch12-03)
    subtitles          parse_srt                                                   (ch12-01, ch12-03)
    display            show_frames                                                 (all demos)

Film: "Tears of Steel" by the Blender Foundation (mango.blender.org), CC BY 3.0, the 720p .mov
(1280 x 534 pixels, 24 frames per second, 17,620 frames, about 370 MB) and the English subtitles
TOS-en.srt. Both are downloaded on first use to data/tears_of_steel/ (ignored by git), and the
results of the slow steps (histograms of all frames, TransNet V2 probabilities) are cached there.
book/figures/ch12_plots.py computes the book's numbers in the same way and shares this cache.

Decoding uses the ffmpeg command-line tool: the system ffmpeg if it is on the PATH, otherwise the
binary of the imageio-ffmpeg package (installed with requirements.txt). Decoding is frame-exact:
frames(t, n) returns frames t to t + n - 1 of the film, counted from 0.

The transitions of 0:00 to 6:00 were annotated by inspecting contact sheets of the decoded frames;
data/ch12/cut_annotation.json holds them (the same file as book/figures/ch12_cut_annotation.json).

Usage:
    from ch12_video import ensure_film, frames, tc, show_frames

    ensure_film()                              # downloads the film and the subtitles once
    strip = frames(5027, 8, step=7)            # 8 RGB frames from 3:29.46 on, every 7th frame
    show_frames(strip, [tc(5027 + 7 * k) for k in range(8)], ncols=4)
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
FILM_DIR = DATA / "tears_of_steel"
FILM = FILM_DIR / "tears_of_steel_720p.mov"
SRT = FILM_DIR / "TOS-en.srt"
ANNOTATION = DATA / "ch12" / "cut_annotation.json"
BASE_URL = "https://download.blender.org/demo/movies/ToS/"

FPS = 24
N_FRAMES = 17620
WIDTH, HEIGHT = 1280, 534
CREDIT = "(CC) Blender Foundation | mango.blender.org, CC BY 3.0"

# scikit-learn's k-means (ch12-01) prints a warning and a traceback on Windows 11, where joblib
# cannot count the physical cores; a core count below the logical count skips that lookup
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

SMALL = (320, 134)       # decoding size for histograms and difference images
TN_SIZE = (48, 27)       # TransNet V2 input size


# ---------------------------------------------------------------- the film

def ensure_film() -> Path:
    """Download the film (about 370 MB) and the English subtitles on first use; return the film path."""
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in [(SRT, BASE_URL + "subtitles/TOS-en.srt"), (FILM, BASE_URL + FILM.name)]:
        if not path.exists():
            print(f"Downloading {url} (once) ...")
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
    return FILM


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise RuntimeError("ffmpeg not found: install it, or pip install imageio-ffmpeg") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(size=SMALL, pix_fmt: str = "gray", start: int = 0, count: int | None = None, step: int = 1):
    """
    Yield frames of the film as uint8 arrays, scaled to `size` (width, height).

    Args:
        size: Output size in pixels, (width, height). The film is 1280 x 534.
        pix_fmt: "gray" (brightness, array height x width) or "rgb24" (array height x width x 3).
        start: Index of the first frame (0-based).
        count: Number of frames to decode (None: up to the end of the film).
        step: Yield every `step`-th frame (all frames are still decoded).
    """
    ensure_film()
    w, h = size
    channels = {"gray": 1, "rgb24": 3}[pix_fmt]
    cmd = [_ffmpeg(), "-v", "error"]
    if start > 0:
        # input seeking is frame-exact: ffmpeg decodes from the previous I-frame and drops the
        # frames before the requested time; a quarter frame earlier avoids rounding to t + 1
        cmd += ["-ss", f"{(start - 0.25) / FPS:.4f}"]
    cmd += ["-i", str(FILM), "-an"]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * channels
    k = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            if k % step == 0:
                a = np.frombuffer(buf, np.uint8)
                yield a.reshape(h, w) if channels == 1 else a.reshape(h, w, channels)
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def frames(start: int, count: int, step: int = 1, size=SMALL, pix_fmt: str = "rgb24") -> np.ndarray:
    """Frames start, start + step, ... (`count` of them) as one array of shape (count, height, width[, 3])."""
    return np.stack(list(decode(size, pix_fmt, start=start, count=(count - 1) * step + 1, step=step)))


def frame(t: int, size=(WIDTH, HEIGHT), pix_fmt: str = "rgb24") -> np.ndarray:
    """A single frame by index, by default in full resolution."""
    return frames(t, 1, size=size, pix_fmt=pix_fmt)[0]


def keyframe_indices() -> np.ndarray:
    """Indices of the frames that the encoder stored as complete images (I-frames)."""
    ensure_film()
    cmd = [_ffmpeg(), "-hide_banner", "-skip_frame", "nokey", "-i", str(FILM), "-an",
           "-vf", "showinfo", "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True).stderr
    return np.array([round(float(x) * FPS) for x in re.findall(r"pts_time:([0-9.]+)", log)])


# ---------------------------------------------------------------- caching and time codes

def cached(name: str, compute):
    """Load data/tears_of_steel/<name>.npy, or compute it with `compute()` and save it there."""
    path = FILM_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)
    result = np.asarray(compute())
    np.save(path, result)
    return result


def tc(t: int) -> str:
    """Frame index to a time code m:ss.ss (minutes, seconds with two decimals)."""
    s = t / FPS
    return f"{int(s // 60)}:{s % 60:05.2f}"


def frame_of(time_code: str) -> int:
    """Time code m:ss or m:ss.ss to the index of the frame shown at that time."""
    m, s = time_code.split(":")
    return int(round((int(m) * 60 + float(s)) * FPS))


# ---------------------------------------------------------------- annotation and evaluation

CUT_TOL = 2   # frames: a detection within two frames of an annotated cut counts as correct


def load_annotation() -> dict:
    """
    The annotated transitions of the training (0:00-3:00) and test (3:00-6:00) excerpts.

    Returns a dict with "train" and "test", each with "first" and "stop" (frame range) and
    "transitions": dicts with "first", "last", "type" ("cut" or "gradual"), and "time". A cut has
    first = last frame of the old shot and last = first frame of the new shot; a gradual
    transition runs from the last clean frame of the old shot to the first clean frame of the new one.
    """
    return json.loads(ANNOTATION.read_text(encoding="utf-8"))["excerpts"]


def evaluate(cuts, graduals, truth, lo: int, hi: int) -> dict:
    """
    Precision, recall, and F1 of detected transitions against the annotation, within frames [lo, hi).

    Args:
        cuts: Frames t with a detected hard cut between t - 1 and t.
        graduals: Detected gradual transitions as (first, last) frame pairs.
        truth: The annotated transitions (load_annotation()[...]["transitions"]).

    A detection matches an annotated transition if the two frame intervals, widened by CUT_TOL
    frames, overlap; every annotated transition is matched at most once. The result lists the
    false detections and the missed transitions as (first, last, type).
    """
    preds = [(t - 1, t, "cut") for t in cuts] + [(r, e, "gradual") for r, e in graduals]
    preds = sorted(p for p in preds if lo <= p[1] < hi)
    true = sorted((x["first"], x["last"], x["type"]) for x in truth if lo <= x["last"] < hi)
    used, tp, false = set(), 0, []
    for p in preds:
        hit = next((k for k, q in enumerate(true)
                    if k not in used and p[0] - CUT_TOL <= q[1] and q[0] <= p[1] + CUT_TOL), None)
        if hit is None:
            false.append(p)
        else:
            used.add(hit)
            tp += 1
    missed = [q for k, q in enumerate(true) if k not in used]
    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(reported=len(preds), annotated=len(true), correct=tp, false=len(false), missed=len(missed),
                precision=precision, recall=recall, f1=f1, false_list=false, missed_list=missed)


# ---------------------------------------------------------------- TransNet V2

def transnet_probabilities() -> tuple[np.ndarray, float]:
    """
    Per-frame transition probabilities of TransNet V2 for the whole film, and the run time in seconds.

    The first call decodes all frames at 48 x 27 pixels and runs the network on the CPU (a few
    minutes); the result is cached in data/tears_of_steel/transnet_v2.npz. Uses the published
    weights of Soucek and Lokoc (2020), packaged as transnetv2-pytorch.
    """
    path = FILM_DIR / "transnet_v2.npz"
    if path.exists():
        z = np.load(path)
        return z["single"], float(z["seconds"])
    import torch
    from transnetv2_pytorch import TransNetV2

    video = np.stack(list(decode(TN_SIZE, "rgb24")))
    model = TransNetV2(device="cpu")
    t0 = time.perf_counter()
    single, many = model.predict_frames(torch.from_numpy(video), quiet=True)
    seconds = time.perf_counter() - t0
    np.savez(path, single=single.numpy(), many=many.numpy(), seconds=seconds)
    return single.numpy(), seconds


def transnet_transitions(p: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Runs of frames with probability above `threshold`, as (first, last) frame of each run.

    TransNet V2 marks the last frame of the old shot: a run (s, s) is a hard cut between s and s + 1.
    """
    on = p > threshold
    runs, t = [], 0
    while t < len(on):
        if on[t]:
            s = t
            while t + 1 < len(on) and on[t + 1]:
                t += 1
            runs.append((s, t))
        t += 1
    return runs


def shots_from_transitions(runs, n: int = N_FRAMES) -> list[tuple[int, int]]:
    """Shots between the TransNet V2 transitions, as (first, last) frame of each shot."""
    shots, start = [], 0
    for s, e in runs:
        if s >= start:
            shots.append((start, s))
        start = e + 1
    if start < n:
        shots.append((start, n - 1))
    return shots


# ---------------------------------------------------------------- the fight sequences

# The three fight sequences of the film, in seconds, identified on contact sheets of one frame per
# second (the same as in the book): the sniper and the flying robot ship, the battle in the church,
# and the robot army.
FIGHTS = [(86, 117), (425, 505), (572, 588)]          # 1:26-1:57, 7:05-8:25, 9:32-9:48


def is_fight(first: int, last: int) -> bool:
    """A shot (first, last frame) counts as a fight shot if at least half of it lies in a fight sequence."""
    a, b = first / FPS, (last + 1) / FPS
    inside = sum(max(0.0, min(b, e) - max(a, s)) for s, e in FIGHTS)
    return inside >= 0.5 * (b - a)


# ---------------------------------------------------------------- motion over the whole film

def film_motion() -> np.ndarray:
    """
    Motion measurements for every frame of the film, at 320 x 134 pixels in gray.

    Returns an array of shape (frames, 3): column 0 is the share of pixels that change by more
    than 20 gray values against the previous frame (frame differencing, ch12-02); columns 1 and 2
    hold, for every 6th frame, the median Lucas-Kanade flow (u, v) in pixels of the small frames
    of up to 200 Shi-Tomasi corners (pyramid with 4 levels, 15 x 15 windows), and NaN otherwise.
    Multiply the flow by 4 for full-size pixels. Cached in data/tears_of_steel/motion_320x134.npy.
    """
    import cv2

    def compute():
        out = np.full((N_FRAMES, 3), np.nan)
        out[0, 0] = 0.0
        prev = None
        for t, g in enumerate(decode(SMALL, "gray")):
            if prev is not None:
                out[t, 0] = (cv2.absdiff(g, prev) > 20).mean()
                if t % 6 == 0:
                    pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8)
                    if pts is not None and len(pts) >= 10:
                        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(15, 15), maxLevel=3)
                        good = st.ravel() == 1
                        if good.sum() >= 10:
                            out[t, 1:] = np.median((nxt - pts).reshape(-1, 2)[good], axis=0)
            prev = g
        return out

    return cached("motion_320x134", compute)


def shot_activity(shots, motion: np.ndarray | None = None) -> np.ndarray:
    """Average share of changed pixels per shot (see film_motion), one value per (first, last) shot."""
    motion = film_motion() if motion is None else motion
    return np.array([np.nanmean(motion[a + 1:b + 1, 0]) if b > a else 0.0 for a, b in shots])


# ---------------------------------------------------------------- RAFT (learned optical flow)

def raft_flow(f0: np.ndarray, f1: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Dense optical flow from RGB frame f0 to f1 with RAFT large (Teed and Deng, 2020) on the CPU.

    Returns the flow as an array of shape (2, height, width), u = flow[0] (to the right) and
    v = flow[1] (downwards) in pixels, and the run time of the network in seconds. The weights
    (torchvision, Raft_Large_Weights.DEFAULT) are downloaded on first use. RAFT needs sides that
    are multiples of 8, so the frames are padded at the bottom and the padding is cut off again.
    """
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).eval()
    h = f0.shape[0]
    pad = (-h) % 8

    def prep(f):
        return torch.from_numpy(np.pad(f, ((0, pad), (0, 0), (0, 0)), mode="edge")).permute(2, 0, 1)[None]

    a, b = weights.transforms()(prep(f0), prep(f1))
    t0 = time.perf_counter()
    with torch.no_grad():
        flow = model(a, b)[-1][0].numpy()[:, :h]
    return flow, time.perf_counter() - t0


# ---------------------------------------------------------------- CLIP (text-to-video search)

CLIP_MODEL = "openai/clip-vit-base-patch32"     # the CLIP model of Chapter 10
CLIP_SIZE = (538, 224)                          # decoding size: short side 224, as the CLIP processor expects


class Clip:
    """CLIP ViT-B/32: images(list of RGB arrays) and text(str) return unit-length 512-value embeddings."""

    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from huggingface_hub.utils import logging as hub_logging
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hub_logging.set_verbosity_error()        # no notice about unauthenticated downloads
        self.torch = torch
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    @staticmethod
    def _tensor(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def images(self, arrays, batch: int = 32) -> np.ndarray:
        from PIL import Image
        out = []
        for i in range(0, len(arrays), batch):
            ims = [Image.fromarray(a) for a in arrays[i:i + batch]]
            with self.torch.no_grad():
                e = self._tensor(self.model.get_image_features(**self.processor(images=ims, return_tensors="pt")))
            e = e.numpy()
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out)

    def text(self, query: str) -> np.ndarray:
        with self.torch.no_grad():
            e = self._tensor(self.model.get_text_features(**self.processor(text=[query], return_tensors="pt",
                                                                            padding=True)))[0].numpy()
        return e / np.linalg.norm(e)


def clip_frame_embeddings(shots, clip: Clip | None = None):
    """
    CLIP embeddings of one frame per second and of the middle frame of every shot.

    Samples frame 12 of every second (the middle of the second). Returns (sampled frame indices,
    their embeddings, middle frame indices, their embeddings). The first call decodes the film once
    and embeds about 900 frames on the CPU (a few minutes); the result is cached in
    data/tears_of_steel/clip_embeddings.npz for these shots.
    """
    path = FILM_DIR / "clip_embeddings.npz"
    sampled = np.arange(FPS // 2, N_FRAMES, FPS)
    middles = np.array([(a + b) // 2 for a, b in shots])
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["middles"], middles):
            return sampled, z["sampled_emb"], middles, z["middle_emb"]
    need = set(sampled.tolist()) | set(middles.tolist())
    picked = {t: f.copy() for t, f in enumerate(decode(CLIP_SIZE, "rgb24")) if t in need}
    clip = clip or Clip()
    sampled_emb = clip.images([picked[t] for t in sampled])
    middle_emb = clip.images([picked[t] for t in middles])
    np.savez(path, sampled_emb=sampled_emb, middles=middles, middle_emb=middle_emb)
    return sampled, sampled_emb, middles, middle_emb


# ---------------------------------------------------------------- subtitles

def parse_srt() -> list[tuple[float, float, str]]:
    """The 76 subtitles of the film as (start seconds, end seconds, text)."""
    ensure_film()
    txt = SRT.read_text(encoding="utf-8-sig", errors="replace").replace("\r", "")
    out = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = block.split("\n")
        g = [int(x) for x in re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1]).groups()]
        out.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                    g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                    re.sub(r"\s+", " ", " ".join(lines[2:])).strip()))
    return out


# ---------------------------------------------------------------- display

def show_frames(images, titles, ncols: int | None = None, width: float = 2.4, cmap=None,
                colors=None) -> None:
    """Show frames side by side (rows of `ncols`), without axes, each with a title.

    `colors` optionally gives a frame color per image (for example green for correct, red for wrong).
    """
    import matplotlib.pyplot as plt

    ncols = ncols or len(images)
    nrows = -(-len(images) // ncols)
    aspect = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * (width * aspect + 0.35)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for k, (ax, img, title) in enumerate(zip(axes.ravel(), images, titles)):
        if cmap and img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if img.dtype != bool and img.max() > 1 else 1)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=8, loc="left")
        if colors is not None and colors[k]:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(colors[k])
                sp.set_linewidth(2.5)
    plt.tight_layout()
    plt.show()
