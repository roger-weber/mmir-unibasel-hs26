"""Media collection for the chapter 9 demos: photos and audio clips under data/.

Sources and licenses are listed in data/ch09/SOURCES.md and data/images/SOURCES.md.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD
from scipy.io import wavfile

DATA = Path("data")

# name -> (file under data/, short description, kind). Pairs of the same kind test whether a
# feature finds "similar content"; red_car and sunset share colors but not content.
IMAGES = {
    "taj_a":      ("ch09/images/taj_a.jpg",      "Taj Mahal, view along the water channel", "Taj Mahal"),
    "taj_b":      ("ch09/images/taj_b.jpg",      "Taj Mahal, 99 seconds later from the garden", "Taj Mahal"),
    "matterhorn": ("images/matterhorn.jpg",      "Matterhorn above Zermatt", "mountain"),
    "mt_forbes":  ("ch09/images/mt_forbes.jpg",  "Mt. Forbes in pink alpenglow", "mountain"),
    "black_cat":  ("ch09/images/black_cat.jpg",  "black cat on a pale quilt", "cat"),
    "ginger_cat": ("ch09/images/ginger_cat.jpg", "ginger kitten in green grass", "cat"),
    "red_car":    ("ch09/images/red_car.jpg",    "red sports car at night", "car"),
    "sunset":     ("ch09/images/sunset.jpg",     "red-orange sunset over the Jura", "sunset"),
    "federer":    ("images/federer.jpg",         "Roger Federer serving at Wimbledon", "tennis player"),
}

# name -> (file under data/, class). Excerpts of 30 s, mono, 16 kHz, 16-bit.
AUDIO = {
    "speech_munsell":      ("ch09/audio/speech_munsell.wav", "speech"),
    "speech_tucker":       ("ch09/audio/speech_tucker.wav", "speech"),
    "speech_hedrick":      ("ch09/audio/speech_hedrick.wav", "speech"),
    "music_morning":       ("ch09/audio/music_morning.wav", "music"),
    "music_mountain_king": ("ch09/audio/music_mountain_king.wav", "music"),
    "music_goldberg_aria": ("ch09/audio/music_goldberg_aria.wav", "music"),
}


def load_images(names=None) -> dict[str, Image.Image]:
    """Load the collection (or the given names) as RGB images, keyed by name."""
    return {n: Image.open(DATA / IMAGES[n][0]).convert("RGB") for n in (names or IMAGES)}


def load_audio(name: str) -> tuple[int, np.ndarray]:
    """Sample rate and samples (float in [-1, 1]) of one audio clip."""
    sr, x = wavfile.read(DATA / AUDIO[name][0])
    return sr, x.astype(np.float32) / 32768


def gps_position(path) -> tuple[float, float] | None:
    """EXIF GPS position of a photo as signed decimal degrees (lat, lon), or None without GPS tags."""
    gps = Image.open(path).getexif().get_ifd(IFD.GPSInfo)
    if 2 not in gps or 4 not in gps:                      # GPSLatitude, GPSLongitude
        return None
    dec = lambda dms, ref: (-1 if ref in ("S", "W") else 1) * sum(float(x) / 60 ** i for i, x in enumerate(dms))
    return dec(gps[2], gps.get(1, "N")), dec(gps[4], gps.get(3, "E"))


def distance_km(p, q, radius=6371.0) -> float:
    """Great-circle (haversine) distance in km between two (lat, lon) points."""
    (lat1, lon1), (lat2, lon2) = (map(math.radians, x) for x in (p, q))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))
