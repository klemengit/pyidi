[![Documentation Status](https://readthedocs.org/projects/pyidi/badge/?version=latest)](https://pyidi.readthedocs.io/en/latest/?badge=latest)
![example workflow](https://github.com/ladisk/pyidi/actions/workflows/python_package_testing.yaml/badge.svg)

# pyIDI

**Image-based Displacement Identification (IDI)** from high-speed video, in Python.

pyIDI reads a recording, tracks the points you select, and returns their sub-pixel
displacement history — ready for modal analysis.

📖 [**Documentation**](https://pyidi.readthedocs.io/en/latest/index.html)

## Installation

```bash
pip install pyidi          # identification
pip install pyidi[qt]      # + the point-selection and result-viewing GUIs
```

Python >= 3.10.

## Quick start

```python
from pyidi import VideoReader, LucasKanade

video = VideoReader('measurement.cih')

lk = LucasKanade(video)
lk.set_points(points=[[150, 200], [150, 260], [150, 320]])   # (row, column)
lk.configure(roi_size=(21, 21))

displacements = lk.get_displacements()   # (n_points, n_frames, 2), in pixels
```

`VideoReader` handles Photron `.cih`/`.cihx`, Phantom `.cine`, Pharsighted `.SLOW`,
image sequences, ordinary video files (MP4, AVI, MOV, ...), and `numpy.ndarray`
stacks of shape `(n_time_points, image_height, image_width)`.

Points are set on the **method** object, not on the `VideoReader`.

### Selecting points interactively

```python
from pyidi import SelectionGUI

gui = SelectionGUI(video, subset_size=21)
lk.set_points(gui)
```

`SelectionGUI` offers a grid inside a polygon, manual points, points along a
polyline, a brush, and automatic filtering onto well-textured image content —
plus vertex dragging and undo. See the
[documentation](https://pyidi.readthedocs.io/en/latest/quick_start/points_selection.html).

<img src="docs/source/quick_start/selection.gif" width="800" />

### Or drive everything from the napari UI

```python
from pyidi import VideoReader, GUI

video = VideoReader('data/data_synthetic.cih')
gui = GUI(video)

displacements = gui.method.displacements
```

<img src="docs/source/quick_start/gifs/napari_full_sof.gif" width="800" />

## Methods

| Method | Solves for | Use it when |
| --- | --- | --- |
| `SimplifiedOpticalFlow` | 2 translations, from the image gradient | a fast first look, motion well below a pixel |
| `LucasKanade` | 2 translations, iteratively | the default choice |
| `DirectionalLucasKanade` | 1 translation along a known direction | motion along a known axis; edge-like features |
| `DIC` | 6 (affine) or 3 (rigid) warp parameters | strain and in-plane rotation, not just translation |

The Lucas-Kanade inner loop is compiled with `numba` and parallelized over points —
one to two orders of magnitude faster than the NumPy implementation.

## Pre-test motion visualization

Eulerian video magnification amplifies subtle, sub-pixel motion directly in the raw
recording, before any identification is run — useful for checking whether and where
a structure moves, and for isolating a single mode:

```python
from pyidi.postprocessing import EulerianMagnifier

evm = EulerianMagnifier(video)
evm.configure(freq_band=(45.0, 55.0), amplification=25)
evm.save('mode_50Hz', output_format='mp4')
```

This is qualitative visualization, **not** a measurement.

## Upgrading

Version 1.0 replaced the monolithic `pyIDI` class with a `VideoReader` plus a
separate method class, so that autocompletion and inline documentation work
properly in VSCode, PyCharm and similar editors. Later releases removed the old
`SubsetSelection` widget and changed how untrackable points are reported.

See the [upgrading guide](https://pyidi.readthedocs.io/en/latest/migration.html)
for what to change. The legacy class is still importable
(`from pyidi import pyIDI`) for compatibility, but is not being developed.

## Developer guidelines

* Add `pyidi/methods/_name_of_method.py` with a class that inherits from `IDIMethod`.
* The class must implement:
  * `configure()` — every parameter stored as a class attribute of the same name
    (this is what makes settings reproducible, picklable and exportable to JSON);
  * `calculate_displacements()` — sets `self.displacements`, of shape
    `(n_points, n_frames, 2)`.
* Export the new class in `pyidi/methods/__init__.py`.

## Citing

If you are using `pyIDI` for your research, consider citing our articles:

- Masmeijer, T., Habtour, E., Zaletelj, K., & Slavič, J. (2024). **Directional DIC method with automatic feature selection**. Mechanical Systems and Signal Processing, 224. https://doi.org/10.1016/j.ymssp.2024.112080
- Čufar, K., Slavič, J., & Boltežar, M. (2024). **Mode-shape magnification in high-speed camera measurements**. Mechanical Systems and Signal Processing, 213, 111336. https://doi.org/10.1016/J.YMSSP.2024.111336
- Zaletelj, K., Gorjup, D., Slavič, J., & Boltežar, M. (2023). **Multi-level curvature-based parametrization and model updating using a 3D full-field response**. Mechanical Systems and Signal Processing, 187, 109927. https://doi.org/10.1016/j.ymssp.2022.109927
- Zaletelj, K., Slavič, J., & Boltežar, M. (2022). **Full-field DIC-based model updating for localized parameter identification**. Mechanical Systems and Signal Processing, 164. https://doi.org/10.1016/j.ymssp.2021.108287
- Gorjup, D., Slavič, J., & Boltežar, M. (2019). **Frequency domain triangulation for full-field 3D operating-deflection-shape identification**. Mechanical Systems and Signal Processing, 133. https://doi.org/10.1016/j.ymssp.2019.106287

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4017153.svg)](https://doi.org/10.5281/zenodo.4017153)
