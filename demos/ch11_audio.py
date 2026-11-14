"""Helpers for the chapter 11 demos (Audio Features): the running clips, framing, and fixed formulas.

Each demo computes its concept step by step in the notebook; this module holds the fixed formulas
and the plumbing that several demos share:

    clips and display     SR, CLIPS, SPEECH_C, MUSIC_C, load, credits, show_spectrogram  (all demos)
    framing               frames, stft_db                                             (all demos)
    frame features        short_time_energy, zero_crossing_rate, spectral_features    (ch11-03; built step
                                                                                       by step in ch11-01)
    segment features      lster, hzcrr, segment_stats                                 (ch11-03; built in ch11-01)
    mel and MFCC          mel, mel_inv, mel_edges, mel_bank, mfcc                      (built in ch11-01)
    pretrained models     Clap (CLAP audio and text encoders), AudioTagger (AST)       (ch11-03)
    speech recognizers    ASR_SR, Whisper, QwenASR                                     (ch11-04)

Audio: librosa example recordings, the same clips as the book chapter. They are downloaded on the
first call of `load()` into the librosa cache (a few MB each; no audio in git). `credits()` lists
titles, authors, and licenses. The parameters (40 ms frames with a hop of 20 ms, 4-second segments,
26 mel bands between 300 and 8000 Hz) follow the book, and the values reproduce its worked examples
(book/figures/ch11_plots.py computes them the same way).

Usage:
    from ch11_audio import SR, load, frames, segment_stats, mfcc

    y = load("libri1")                       # float32 samples at 22,050 Hz, mono
    F, n = frames(y)                         # (741, 882): one row per 40 ms frame
    seg = segment_stats(y)                   # (11, 2): LSTER and HZCRR per 4-second segment
"""

import contextlib
import io

import numpy as np

SR = 22050                         # sampling rate of the running clips (Hz)
FRAME_MS, HOP_MS = 40, 20          # frames of the perceptual features
SEG_S, SEG_HOP_S = 4.0, 1.0        # segments for LSTER and HZCRR
SPEECH_C, MUSIC_C = "#2a6fdb", "#d9822b"   # plot colors of the book figures

# librosa example name -> (short title, kind). Kind is "speech", "music", or "animal".
CLIPS = {
    "libri1":     ("LibriVox reading 1 (The Ashiel Mystery)", "speech"),
    "libri2":     ("LibriVox reading 2 (The Age of Chivalry)", "speech"),
    "libri3":     ("LibriVox reading 3 (Northanger Abbey)", "speech"),
    "brahms":     ("Brahms, Hungarian Dance No. 5", "music"),
    "nutcracker": ("Tchaikovsky, Dance of the Sugar Plum Fairy", "music"),
    "vibeace":    ("Vibe Ace (jazz)", "music"),
    "trumpet":    ("Trumpet loop (90 bpm)", "music"),
    "robin":      ("Robin call", "animal"),
    "humpback":   ("Humpback whale, Glacier Bay", "animal"),
}

# librosa's metadata names libri3 "Sense and Sensibility, chapter 18", but the LibriSpeech transcript
# of utterance 198-209-0000 (Catherine, Mrs. Allen, Miss Tilney) is from Jane Austen's Northanger Abbey.
TITLE_FIXES = {"libri3": "Northanger Abbey (librosa metadata: Sense and Sensibility, chapter 18)"}

_cache: dict = {}


def load(name: str, sr: int = SR) -> np.ndarray:
    """Samples of a running clip as float32, mono, resampled to `sr` (downloaded on first use)."""
    import librosa
    key = (name, sr)
    if key not in _cache:
        _cache[key], _ = librosa.load(librosa.ex(name), sr=sr, mono=True)
    return _cache[key]


def credits(names=None) -> list[list[str]]:
    """Rows [clip, title, attribution, license] for the clips in `names` (default: all)."""
    import librosa
    rows = []
    for name in names or CLIPS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            librosa.util.example_info(name)
        lines = buf.getvalue().splitlines()
        meta = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:]) if v}
        title = TITLE_FIXES.get(name, lines[0].strip())
        rows.append([name, title, meta.get("attribution", ""), meta.get("license", "")])
    return rows


# ---------------------------------------------------------------- framing and spectrogram

def frames(y: np.ndarray, sr: int = SR, frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS):
    """Cut `y` into overlapping frames. Returns (F, N): F has one row of N samples per frame."""
    n = int(round(sr * frame_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    count = 1 + (len(y) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return y[idx], n


def stft_db(y: np.ndarray, win_ms: float, hop_ms: float, sr: int = SR):
    """Power spectrogram in dB with a Hann window. Returns (S_db, n, hop); S_db is bins x frames."""
    import librosa
    n = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    S = np.abs(librosa.stft(y, n_fft=n, hop_length=hop, win_length=n, window="hann")) ** 2
    return 10 * np.log10(S + 1e-10), n, hop


def show_spectrogram(ax, S_db, n, hop, sr: int = SR, t0: float = 0.0, f_max: float | None = None,
                     dyn_db: float = 80, title: str | None = None):
    """Draw a dB spectrogram (bins x frames) on `ax`: time to the right, frequency upwards."""
    f = np.arange(S_db.shape[0]) * sr / n
    t = t0 + np.arange(S_db.shape[1]) * hop / sr
    top = S_db.max()
    mesh = ax.pcolormesh(t, f, S_db, shading="nearest", cmap="magma", vmin=top - dyn_db, vmax=top)
    ax.set_ylim(0, f_max or sr / 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    return mesh


# ---------------------------------------------------------------- frame and segment features

def short_time_energy(F: np.ndarray) -> np.ndarray:
    """Linear short-time energy E(t): the mean squared amplitude of each frame."""
    return np.mean(F ** 2, axis=1)


def zero_crossing_rate(F: np.ndarray) -> np.ndarray:
    """Zero-crossing rate Z(t): sign changes per frame, normalized by 2N (zero counts as positive)."""
    s = np.sign(F)
    s[s == 0] = 1
    return np.sum(np.abs(np.diff(s, axis=1)), axis=1) / (2 * F.shape[1])


def spectra(F: np.ndarray, window: str = "hann") -> np.ndarray:
    """Magnitude spectra |X(t,k)| for k = 0..N/2 of the windowed frames."""
    w = np.hanning(F.shape[1]) if window == "hann" else np.hamming(F.shape[1])
    return np.abs(np.fft.rfft(F * w, axis=1))


def spectral_features(X: np.ndarray, n: int, sr: int = SR) -> dict[str, np.ndarray]:
    """Centroid, roll-off (85%), bandwidth, and flux of magnitude spectra X (frames x bins)."""
    freqs = np.arange(X.shape[1]) * sr / n
    tot = X.sum(axis=1) + 1e-12
    sc = (X * freqs).sum(axis=1) / tot
    rolloff = freqs[np.argmax(np.cumsum(X, axis=1) >= 0.85 * tot[:, None], axis=1)]
    sb = np.sqrt((X * (freqs[None, :] - sc[:, None]) ** 2).sum(axis=1) / tot)
    P = X / tot[:, None]
    sf = np.r_[np.nan, np.sum(np.diff(P, axis=0) ** 2, axis=1)]
    return {"centroid": sc, "rolloff": rolloff, "bandwidth": sb, "flux": sf}


def lster(ste: np.ndarray) -> float:
    """Low short-time energy ratio: share of frames below half the mean (linear) energy."""
    return float(np.mean(ste < 0.5 * ste.mean()))


def hzcrr(zcr: np.ndarray) -> float:
    """High zero-crossing rate ratio: share of frames above 1.5 times the mean ZCR."""
    return float(np.mean(zcr > 1.5 * zcr.mean()))


def segment_stats(y: np.ndarray, seg_s: float = SEG_S, hop_s: float = SEG_HOP_S) -> np.ndarray:
    """LSTER and HZCRR of every segment of `seg_s` seconds, `hop_s` apart. Shape (segments, 2)."""
    F, _ = frames(y)
    ste, zcr = short_time_energy(F), zero_crossing_rate(F)
    per_seg = int(round((seg_s * 1000 - FRAME_MS) / HOP_MS)) + 1      # 199 frames for 4 s
    step = int(round(hop_s * 1000 / HOP_MS))
    return np.array([(lster(ste[a:a + per_seg]), hzcrr(zcr[a:a + per_seg]))
                     for a in range(0, len(ste) - per_seg + 1, step)])


# ---------------------------------------------------------------- mel scale and MFCC

def mel(f):
    """mel(f) = 1125 ln(1 + f/700)."""
    return 1125 * np.log(1 + np.asarray(f, float) / 700)


def mel_inv(m):
    """f(mel) = 700 (e^(mel/1125) - 1)."""
    return 700 * (np.exp(np.asarray(m, float) / 1125) - 1)


def mel_edges(B: int = 26, f_lo: float = 300, f_hi: float = 8000) -> np.ndarray:
    """The B + 2 band edges f_c(0..B+1) in Hz, equally spaced in mel."""
    return mel_inv(mel(f_lo) + np.arange(B + 2) * (mel(f_hi) - mel(f_lo)) / (B + 1))


def mel_bank(n: int, sr: int = SR, B: int = 26, f_lo: float = 300, f_hi: float = 8000) -> np.ndarray:
    """Triangular weights d(b,k) of B mel bands for the bins of an n-sample frame. Shape (B, n//2 + 1)."""
    fc = mel_edges(B, f_lo, f_hi)
    f = np.arange(n // 2 + 1) * sr / n
    D = np.zeros((B, len(f)))
    for b in range(1, B + 1):
        up = (f - fc[b - 1]) / (fc[b] - fc[b - 1])
        down = (fc[b + 1] - f) / (fc[b + 1] - fc[b])
        D[b - 1] = np.clip(np.minimum(up, down), 0, None)
    return D


def mfcc(y: np.ndarray, sr: int = SR, B: int = 26, n_coef: int = 13, deltas: bool = True) -> np.ndarray:
    """MFCC per 40 ms frame as in the book: Hamming window, power spectrum, B mel bands,
    natural log, DCT-II. Returns (frames, 39) with deltas, else (frames, n_coef)."""
    F, n = frames(y, sr)
    M = (spectra(F, "hamming") ** 2) @ mel_bank(n, sr, B).T            # M(t,b)
    j = np.arange(n_coef)[:, None]
    b = np.arange(1, B + 1)[None, :]
    c = np.log(M + 1e-10) @ np.cos(j * (2 * b - 1) * np.pi / (2 * B)).T
    if not deltas:
        return c
    d = np.zeros_like(c)
    d[1:-1] = (c[2:] - c[:-2]) / 2
    dd = np.zeros_like(c)
    dd[1:-1] = (d[2:] - d[:-2]) / 2
    return np.hstack([c, d, dd])


# ---------------------------------------------------------------- pretrained audio models (ch11-03)

CLAP_ID = "laion/clap-htsat-unfused"                      # CLAP, audio at 48 kHz
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"       # Audio Spectrogram Transformer on AudioSet, 16 kHz


@contextlib.contextmanager
def _quiet_loading():
    """Hide download progress bars, Hub notices, and harmless loader warnings in notebook output."""
    import logging
    import warnings
    import transformers
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class Clap:
    """CLAP audio and text encoders (downloaded on first use, about 600 MB; CPU is enough).

    audio(clips) and text(prompts) return L2-normalized embeddings (one row each), so a dot
    product is the cosine similarity. scale is the learned temperature that turns similarities
    into the logits of the model's softmax."""

    SR = 48000

    def __init__(self, model_id: str = CLAP_ID):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        with _quiet_loading():
            self.proc = ClapProcessor.from_pretrained(model_id)
            self.model = ClapModel.from_pretrained(model_id).eval()
        self.scale = float(self.model.logit_scale_a.detach().exp())

    def _norm(self, x):
        return (x / x.norm(dim=-1, keepdim=True)).numpy()

    def audio(self, clips) -> np.ndarray:
        inp = self.proc(audio=list(clips), sampling_rate=self.SR, return_tensors="pt")
        with self.torch.no_grad():
            return self._norm(self.model.get_audio_features(**inp).pooler_output)

    def text(self, prompts) -> np.ndarray:
        inp = self.proc(text=list(prompts), return_tensors="pt", padding=True)
        with self.torch.no_grad():
            return self._norm(self.model.get_text_features(**inp).pooler_output)


class AudioTagger:
    """AST classifier for the 527 AudioSet classes (downloaded on first use, about 350 MB)."""

    SR = 16000

    def __init__(self, model_id: str = AST_ID):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        with _quiet_loading():
            self.fe = ASTFeatureExtractor.from_pretrained(model_id)
            self.model = ASTForAudioClassification.from_pretrained(model_id).eval()
        self.labels = self.model.config.id2label

    def features(self, y: np.ndarray) -> np.ndarray:
        """The model input: a log mel spectrogram of 1024 frames (10 ms) x 128 mel bands."""
        return self.fe(y, sampling_rate=self.SR, return_tensors="np")["input_values"][0]

    def scores(self, y: np.ndarray) -> np.ndarray:
        """One independent score (sigmoid) per AudioSet class."""
        inp = self.fe(y, sampling_rate=self.SR, return_tensors="pt")
        with self.torch.no_grad():
            return self.model(**inp).logits.sigmoid()[0].numpy()

    def top(self, y: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
        s = self.scores(y)
        return [(self.labels[int(i)], float(s[i])) for i in np.argsort(s)[::-1][:k]]


# ---------------------------------------------------------------- speech recognizers (ch11-04)

WHISPER_ID = "openai/whisper-small"                      # 244M parameters, about 1 GB
QWEN_ASR_ID = "Qwen/Qwen3-ASR-0.6B-hf"                   # 0.6B (0.78B with the audio encoder), about 1.6 GB
ASR_SR = 16000                                           # both recognizers expect 16 kHz


class Whisper:
    """Whisper through transformers (downloaded on first use). Input: float samples at 16 kHz."""

    def __init__(self, model_id: str = WHISPER_ID):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
        self.torch = torch
        with _quiet_loading():
            self.proc = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
            self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.proc.tokenizer,
                                 feature_extractor=self.proc.feature_extractor, device="cpu")

    def transcribe(self, y: np.ndarray, language: str | None = None, prompt: str | None = None) -> str:
        """Transcript of up to 30 s of audio. `prompt` is text the decoder sees as previous context,
        for example expected spellings of names; `language` forces a language code such as "de"."""
        feats = self.proc(y, sampling_rate=ASR_SR, return_tensors="pt").input_features
        # Decode with timestamps, as words() does, so that both return the same transcript.
        kwargs = {"language": language, "task": "transcribe", "return_timestamps": True}
        if prompt:
            kwargs["prompt_ids"] = self.proc.get_prompt_ids(prompt, return_tensors="pt")
        with self.torch.no_grad(), _quiet_loading():
            ids = self.model.generate(feats, **kwargs)
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text[len(prompt):].strip() if prompt and text.startswith(prompt) else text

    def words(self, y: np.ndarray, language: str | None = None) -> list[tuple[str, float, float]]:
        """Transcript as (word, start, end) with times in seconds."""
        with _quiet_loading():
            out = self.pipe({"raw": y, "sampling_rate": ASR_SR}, return_timestamps="word",
                            generate_kwargs={"language": language} if language else {})
        return [(c["text"].strip(), *c["timestamp"]) for c in out["chunks"]]

    def language_probs(self, y: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """The most probable spoken languages: the decoder's first prediction after <|startoftranscript|>."""
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        tok = self.proc.tokenizer
        codes = [f"<|{c}|>" for c in LANGUAGES]
        ids = tok.convert_tokens_to_ids(codes)
        feats = self.proc(y, sampling_rate=ASR_SR, return_tensors="pt").input_features
        start = self.torch.tensor([[tok.convert_tokens_to_ids("<|startoftranscript|>")]])
        with self.torch.no_grad():
            logits = self.model(input_features=feats, decoder_input_ids=start).logits[0, -1, ids]
        p = logits.softmax(-1).numpy()
        return [(codes[i].strip("<|>"), float(p[i])) for i in np.argsort(p)[::-1][:top]]


class QwenASR:
    """Qwen3-ASR, an LLM-based recognizer, through transformers (downloaded on first use)."""

    def __init__(self, model_id: str = QWEN_ASR_ID):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        self.torch = torch
        with _quiet_loading():
            self.proc = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).eval()

    def transcribe(self, y: np.ndarray, language: str | None = None, prompt: str | None = None) -> dict:
        """{"language": ..., "transcription": ...}. `prompt` is text context such as a list of
        hotwords; `language` forces a language (name or code), else the model identifies it."""
        inp = self.proc.apply_transcription_request(audio=y, language=language, prompt=prompt)
        with self.torch.no_grad(), _quiet_loading():
            out = self.model.generate(**inp, max_new_tokens=400)
        res = self.proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="parsed")
        res = res[0] if isinstance(res, list) else res
        if language and not res.get("language"):
            res["language"] = language
        return res
