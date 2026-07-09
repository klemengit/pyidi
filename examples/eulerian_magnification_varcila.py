# %%

"""
Eulerian Video Magnification (EVM) of a real Varcila high-speed recording, tuned to
an *identified* structural mode.

Context (from the project's analysis notebooks in Varcila_high_speed/):
- The true capture rate is fps = 60 (the mp4 container reports a meaningless 30).
- Displacement identification found two low-frequency modes; the exported mode-shape
  gifs (Varcila_shape1/2.gif) correspond to peaks at ~0.375 Hz (band 0.30-0.45 Hz)
  and ~0.75 Hz (band 0.70-0.80 Hz).

This script amplifies the ~0.375 Hz mode directly from the raw video, as a
qualitative reveal (NOT a measurement). Because the mode is sub-1 Hz, we need many
frames (several ~2.7 s periods), so we stream frames in and spatially downscale to
keep memory bounded.
"""

import os

import numpy as np
import cv2
import imageio.v3 as iio

import pyidi
from pyidi.postprocessing import EulerianMagnifier

# %%
# ---- Configuration -----------------------------------------------------------

VIDEO_PATH = (
    "/home/klemenzaletelj/Data/Projekti/_Arhiv/Varcila_high_speed/video.mp4"
)
FPS = 60                  # true capture rate (from the analysis notebooks)
N_FRAMES = 1200           # ~20 s -> ~7 periods of the 0.375 Hz mode
DOWNSCALE = 3             # spatial factor: 1920x1080 -> 640x360 (bounds memory)

# --- pick the mode to amplify ---
# FREQ_BAND = (0.30, 0.45)  # ~0.375 Hz mode (Varcila_shape1)
FREQ_BAND = (0.70, 0.80)  # ~0.75 Hz mode (Varcila_shape2)

AMPLIFICATION = 30        # alpha; motion is sub-pixel + downscaled, so amplify hard
LEVELS = 4
FILTER_TYPE = "ideal"
OUTPUT_STEM = os.path.join(os.path.dirname(__file__), "evm_varcila_mode")

# %%
# ---- Stream frames in and downscale (avoids a multi-GB full-res load) ---------

probe = pyidi.VideoReader(VIDEO_PATH)
h = probe.image_height // DOWNSCALE
w = probe.image_width // DOWNSCALE
print(f"source: {probe.N} frames @ {probe.image_height}x{probe.image_width}; "
      f"using {N_FRAMES} frames downscaled to {h}x{w}")

small = np.empty((N_FRAMES, h, w), dtype=np.uint8)
for i, frame in enumerate(iio.imiter(VIDEO_PATH, plugin="pyav")):
    if i >= N_FRAMES:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    small[i] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
print(f"loaded stack: {small.shape}, {small.nbytes / 1e6:.0f} MB")

# %%
# ---- Magnify the chosen mode -------------------------------------------------

root = os.path.join(os.path.dirname(OUTPUT_STEM), "evm_tmp_root")
video = pyidi.VideoReader(small, root=root, fps=FPS)

# Optional: restrict amplification to a region of interest so the background /
# whole frame does not appear to move. The mask must match the (downscaled) frame
# size (h, w). A soft (feathered) edge avoids seams. Example rectangular ROI:
#
# mask = np.zeros((h, w), dtype=np.float32)
# mask[80:280, 200:460] = 1.0
# mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)  # feather the edges
#
# then pass mask=mask below.

mag = EulerianMagnifier(video)
mag.configure(
    freq_band=FREQ_BAND,
    amplification=AMPLIFICATION,
    levels=LEVELS,
    filter_type=FILTER_TYPE,
    # mask=mask,          # <- uncomment to magnify only inside the ROI
    show_progress=True,   # progress bars for pyramid / filtering / reconstruction
)
print(f"magnifying band {FREQ_BAND} Hz at alpha={AMPLIFICATION} ...")
magnified = mag.get_magnified_video()
print(f"magnified: {magnified.shape}, dtype {magnified.dtype}")

# %%
# ---- Save --------------------------------------------------------------------

mag.save(OUTPUT_STEM, output_format="mp4")
print(f"Saved: {OUTPUT_STEM}.mp4")

# A gif of all 1200 frames is huge (~180 MB). For a shareable clip, save a short
# window instead, e.g. one period (~160 frames at 60 fps):
# mag.save(OUTPUT_STEM, output_format="gif", frame_range=(0, 160))

# %%
# ---- (optional) verify the amplified motion sits in the target band ----------

fps_v = FPS
spec = np.abs(np.fft.rfft(magnified.astype(np.float32)
                          - magnified.mean(0, keepdims=True), axis=0)).mean(axis=(1, 2))
fvec = np.fft.rfftfreq(magnified.shape[0], d=1 / fps_v)
peak = np.argmax(spec[1:]) + 1
print(f"dominant temporal frequency of magnified motion: {fvec[peak]:.3f} Hz "
      f"(target band {FREQ_BAND} Hz)")
