"""Helpers for the chapter 10 demos (Image Features): photos, color conversions, and plumbing.

Each demo computes its concept step by step in the notebook; this module holds the fixed formulas
and the plumbing that several demos share:

    photos and display     PHOTOS, load_photos, thumbnail, crop_32, show_images      (all demos)
    color                  rgb_to_lab, lab_to_rgb                                     (all demos)
    color histogram        lab_histogram, lab_histogram_fast, lab_similarity_matrix, quadratic  (ch10-02, ch10-04;
                           built step by step in ch10-01)
    gradients              SOBEL_X, SOBEL_Y, lightness, gradients                     (ch10-02, ch10-03)
    collection             load_collection, load_image                                (ch10-02 to ch10-04)
    SIFT                   sift_features, ratio_matches, consistent_matches, sift_score  (ch10-03, ch10-04)
    neural embeddings      Embedders (ResNet-18, CLIP ViT-B/32, DINOv2 ViT-S/14)      (ch10-04)

The three photos of section 10.1 are committed under data/images/; data/images/SOURCES.md lists
authors and licenses. The 43 other photos of the collection are downloaded from Wikimedia Commons
on first use (data/images/ch10_collection.json lists authors and licenses). The color conversions
use the sRGB matrix of IEC 61966-2-1 and the D65 white point, as in appendix E.2 of the book, and
reproduce the book's worked-example values (book/figures/ch10_plots.py computes them the same way).

Usage:
    from ch10_features import PHOTOS, load_photos, rgb_to_lab, thumbnail, show_images

    photos = load_photos()                   # {"matterhorn": uint8 array (H, W, 3), ...}
    lab = rgb_to_lab(photos["matterhorn"])   # float array (H, W, 3): L, a, b
    show_images([thumbnail(p) for p in photos.values()], list(photos))
"""

from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).parent / "data"

# name -> (file under data/, display title)
PHOTOS = {
    "matterhorn": ("images/matterhorn.jpg", "Matterhorn"),
    "taj_mahal": ("images/taj_mahal.jpg", "Taj Mahal"),
    "federer": ("images/federer.jpg", "Roger Federer"),
}


def load_photos(names=None) -> dict[str, np.ndarray]:
    """sRGB pixel arrays (height x width x 3, uint8) of the photos, in the order of PHOTOS."""
    return {n: np.asarray(Image.open(DATA / PHOTOS[n][0]).convert("RGB")) for n in (names or PHOTOS)}


# ---------------------------------------------------------------- color conversions

# sRGB (linear) to CIE XYZ, IEC 61966-2-1, and the D65 white point as the XYZ of sRGB white
M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]])
WHITE_D65 = M_RGB2XYZ @ np.ones(3)


def srgb_to_linear(rgb) -> np.ndarray:
    """Undo the sRGB gamma encoding: 0..255 to linear light 0..1."""
    c = np.asarray(rgb, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _f(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def rgb_to_lab(rgb, linearize: bool = True) -> np.ndarray:
    """
    sRGB (0..255, channels on the last axis) to CIE Lab under D65.

    Args:
        rgb: Array of shape (..., 3) with values 0..255, for example an image or a list of colors.
        linearize: Undo the gamma encoding first (correct). False skips this step and treats the
            gamma-encoded values as linear light, the mistake that the book warns about.

    Returns:
        Float array of the same shape: L (0..100), a (green to red), b (blue to yellow).
    """
    lin = srgb_to_linear(rgb) if linearize else np.asarray(rgb, float) / 255.0
    xyz = lin @ M_RGB2XYZ.T / WHITE_D65
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def lab_to_rgb(lab) -> np.ndarray:
    """CIE Lab (D65) to sRGB 0..1, clipped to the displayable range; the inverse of rgb_to_lab."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    d = 6 / 29
    xyz = np.stack([np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29)) for t in (fx, fy, fz)], -1) * WHITE_D65
    lin = np.clip(xyz @ np.linalg.inv(M_RGB2XYZ).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- color histogram (built step by step in ch10-01)

def lab_histogram(rgb, step: int = 20) -> np.ndarray:
    """Share of the pixels per Lab reference color: ranges of `step` units on L (0 to 100) and on a and
    b (-50 to 50), outer ranges open; 125 reference colors for step 20, ordered by L, then a, then b."""
    return lab_histogram_of_lab(rgb_to_lab(rgb), step)


def lab_histogram_fast(rgb, step: int = 20) -> np.ndarray:
    """The same histogram with OpenCV's color conversion, which is much faster than the NumPy formulas
    and gives nearly the same Lab values (used for the cost comparison in ch10-04)."""
    import cv2

    return lab_histogram_of_lab(cv2.cvtColor((np.asarray(rgb) / 255).astype(np.float32), cv2.COLOR_RGB2Lab), step)


def lab_histogram_of_lab(lab, step: int = 20) -> np.ndarray:
    """The histogram of lab_histogram for pixels that are already in Lab."""
    n = 100 // step
    L = np.clip(lab[..., 0] // step, 0, n - 1).astype(int)
    a = np.clip((lab[..., 1] + 50) // step, 0, n - 1).astype(int)
    b = np.clip((lab[..., 2] + 50) // step, 0, n - 1).astype(int)
    h = np.bincount(((L * n + a) * n + b).ravel(), minlength=n ** 3).astype(float)
    return h / h.sum()


def lab_similarity_matrix(step: int = 20) -> np.ndarray:
    """a_ij = 1 - d_ij / max d between the Lab centers of the reference colors of lab_histogram."""
    n = 100 // step
    cL = (np.arange(n) + 0.5) * step
    cab = -50 + (np.arange(n) + 0.5) * step
    centers = np.array([(l, a, b) for l in cL for a in cab for b in cab])
    D = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    return 1 - D / D.max()


def quadratic(h, g, A) -> float:
    """Quadratic histogram distance (h - g)^T A (h - g)."""
    d = h - g
    return float(d @ A @ d)


# ---------------------------------------------------------------- gradients (section 10.2)

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0


def lightness(rgb) -> np.ndarray:
    """The lightness channel L of CIE Lab (0 to 100) as float32, the input of the edge features."""
    return rgb_to_lab(rgb)[..., 0].astype(np.float32)


def gradients(L, smooth_sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Slide the Sobel kernels over every pixel (a weighted sum, mirrored at the border) and return g_x
    and g_y. With smooth_sigma, the image is first smoothed with a Gaussian of that width."""
    import cv2

    L = np.asarray(L, np.float32)
    if smooth_sigma:
        L = cv2.GaussianBlur(L, (0, 0), smooth_sigma)
    gx = cv2.filter2D(L, -1, SOBEL_X, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(L, -1, SOBEL_Y, borderType=cv2.BORDER_REFLECT)
    return gx, gy


# ---------------------------------------------------------------- the 45-photo collection (sections 10.2 to 10.4)

def load_image(path, longest: int = 1280) -> Image.Image:
    """Open a photo as RGB, scaled down (Lanczos) so that its longest side is at most `longest` pixels."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > longest:
        s = longest / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def load_collection() -> list[dict]:
    """
    The chapter 10 collection: 43 Wikimedia Commons photos (downloaded on first use, about 16 MB) plus
    the committed Taj Mahal and Federer photos, 45 items in the book's order.

    Each item: {"id", "group", "is_matterhorn", "is_mountain", "description", "tags", "path"}, where
    group is "matterhorn" (8), "mountain" (12 other peaks), "color_decoy" (9 scenes with the query's
    palette but no mountain), or "other" (16).
    """
    from shared.collections.images import load_image_collection

    keys = ("id", "group", "is_matterhorn", "is_mountain", "description", "tags", "path")
    items = [{k: e[k] for k in keys} for e in load_image_collection()]
    for n in ("taj_mahal", "federer"):
        items.append(dict(id=n, group="other", is_matterhorn=False, is_mountain=False,
                          description=PHOTOS[n][1], tags=[], path=DATA / PHOTOS[n][0]))
    return items


# ---------------------------------------------------------------- SIFT (section 10.2, demonstrated step by step in ch10-03)

def sift_features(rgb):
    """SIFT keypoints and their 128-dimensional descriptors on the gray version of an RGB image."""
    import cv2

    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.SIFT_create().detectAndCompute(gray, None)


def ratio_matches(des_a, des_b, ratio: float = 0.8) -> list:
    """Nearest-neighbor matches from a to b that pass the ratio test (nearest < ratio x second nearest)."""
    import cv2

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def consistent_matches(kp_a, kp_b, matches):
    """The geometric check: RANSAC finds the rotation, scaling, and shift that most matches agree on
    (within 5 pixels). Returns the 2 x 3 transformation (or None) and a boolean mask of the agreeing matches."""
    import cv2

    if len(matches) < 3:
        return None, np.zeros(len(matches), bool)
    src = np.float32([kp_a[m.queryIdx].pt for m in matches])
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M, (np.zeros(len(matches), bool) if inliers is None else inliers.ravel().astype(bool))


def sift_score(a, b) -> tuple[int, int]:
    """For two (keypoints, descriptors) pairs: the number of ratio-test matches and of consistent matches."""
    m = ratio_matches(a[1], b[1])
    _, ok = consistent_matches(a[0], b[0], m)
    return len(m), int(ok.sum())


# ---------------------------------------------------------------- neural embeddings (sections 10.3 and 10.4)

class Embedders:
    """
    The three pretrained models of the chapter, loaded from Hugging Face (about 0.6 GB on the first run,
    CPU is enough). All embeddings are returned as float64 vectors of unit length.

        resnet(im, stage)   ResNet-18 (microsoft/resnet-18): feature maps of a stage, averaged (512 values
                            for the last stage)
        resnet_top(im, k)   the k most probable ImageNet classes with their probabilities
        clip_image(im)      CLIP ViT-B/32 (openai/clip-vit-base-patch32) image embedding (512 values)
        clip_text(text)     CLIP text embedding in the same space (512 values)
        dino_image(im)      DINOv2 ViT-S/14 (facebook/dinov2-small) [class] token embedding (384 values)

    `im` is a PIL image. The models themselves are the attributes resnet_model, clip_model, dino_model.
    """

    def __init__(self):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from transformers import (AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor,
                                  ResNetForImageClassification)
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

        self.torch = torch
        # use_fast=False: the PIL-based image processors. With torchvision installed, transformers would
        # otherwise pick the "fast" torch-based ones, whose resizing shifts the similarities in the third or
        # fourth decimal; the book's values were computed with the PIL-based processors.
        self.resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").eval()
        self.resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18", use_fast=False)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=False)

    @staticmethod
    def _unit(v) -> np.ndarray:
        v = v.detach().numpy().astype(np.float64).ravel()
        return v / np.linalg.norm(v)

    @staticmethod
    def _tensor(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def resnet_outputs(self, im):
        """The full ResNet-18 output with the feature maps of every stage (hidden_states) and the logits."""
        with self.torch.no_grad():
            return self.resnet_model(**self.resnet_processor(images=im, return_tensors="pt"), output_hidden_states=True)

    def resnet(self, im, stage: int = -1) -> np.ndarray:
        """Global average pooling of the feature maps of one stage (1 to 4, or -1 for the last)."""
        return self._unit(self.resnet_outputs(im).hidden_states[stage].mean(dim=(2, 3))[0])

    def resnet_top(self, im, k: int = 2) -> list[tuple[str, float]]:
        p = self.resnet_outputs(im).logits.softmax(-1)[0]
        top = p.topk(k)
        return [(self.resnet_model.config.id2label[i.item()], v.item()) for v, i in zip(top.values, top.indices)]

    def clip_image(self, im) -> np.ndarray:
        with self.torch.no_grad():
            out = self.clip_model.get_image_features(**self.clip_processor(images=im, return_tensors="pt"))
        return self._unit(self._tensor(out)[0])

    def clip_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            out = self.clip_model.get_text_features(**self.clip_processor(text=[text], return_tensors="pt", padding=True))
        return self._unit(self._tensor(out)[0])

    def dino_image(self, im) -> np.ndarray:
        with self.torch.no_grad():
            return self._unit(self.dino_model(**self.dino_processor(images=im, return_tensors="pt")).pooler_output[0])


# ---------------------------------------------------------------- display

def thumbnail(rgb: np.ndarray, longest: int = 480) -> np.ndarray:
    """A smaller copy for display (Lanczos filter); features are computed on the full photo."""
    im = Image.fromarray(np.asarray(rgb))
    im.thumbnail((longest, longest), Image.LANCZOS)
    return np.asarray(im)


def crop_32(im, size=(300, 200)) -> np.ndarray:
    """Center crop to 3:2 and resize, so that thumbnails of different shapes line up in a grid."""
    im = im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
    w, h = im.size
    if w / h > 1.5:
        nw = int(h * 1.5)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / 1.5)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return np.asarray(im.resize(size, Image.LANCZOS))


def show_images(images, titles, ncols: int | None = None, height: float = 2.6) -> None:
    """Show images side by side (in rows of `ncols`), without axes, each with a title."""
    import matplotlib.pyplot as plt

    ncols = ncols or len(images)
    nrows = -(-len(images) // ncols)
    aspect = images[0].shape[1] / images[0].shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height * aspect, nrows * (height + 0.4)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, loc="left")
    plt.tight_layout()
    plt.show()
