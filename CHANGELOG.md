# Changelog

This changelog starts at version 1.4.0. For earlier versions see the
[commit history](https://github.com/ladisk/pyidi/commits/master).

## Unreleased

### Eulerian video magnification

New `EulerianMagnifier` class in `pyidi.postprocessing` (also available as a
functional `eulerian_magnification()` wrapper) adds linear Eulerian Video
Magnification (Wu et al., SIGGRAPH 2012) as a pre-test visualization tool: a
Laplacian pyramid decomposition, a temporal band-pass filter applied per
pyramid level, and linear amplification of the band-passed signal added back
onto the original. It reveals subtle, often sub-pixel motion directly in the
raw recording, before any displacement identification is run, so it is useful
for checking whether (and where) a structure is moving and for picking
regions of interest or seed points ahead of a full analysis. **This is
qualitative visualization only, not a measurement** - the amplification
distorts motion amplitudes non-linearly and must not be read as displacement.

Configure with `freq_band=(low, high)` in Hz to isolate a suspected mode,
`amplification` for the gain, and `levels` for the pyramid depth. The
temporal filter is `filter_type="ideal"` (FFT brick-wall, default) or
`"butter"` (Butterworth). An optional 2D `mask` restricts amplification to a
region of interest, leaving the rest of the frame as recorded. `save()`
writes the result to mp4/avi/mov/gif, mapping the intensity range to 8-bit
for playback.

The optional `lambda_c` spatial-wavelength attenuation, meant to damp
amplification of fine, noisy detail while leaving broad structural motion at
full gain, initially ramped in the wrong direction: the finest pyramid level
got the strongest amplification and the coarsest the weakest, the reverse of
what Wu et al. specify. This is now fixed to ramp from the coarsest band down
to the finest. A warning is also now raised if `lambda_c` ends up attenuating
every level to zero (the output would then equal the input unchanged), and
`save()` raises rather than silently defaulting to 30 fps when no frame rate
is available. The test suite was substantially hardened alongside these
fixes, including mutation-verified tests that would have caught a disabled
band-pass, a sign-inverted amplification, or a dropped pyramid band.

### Rigid body motion in `DirectionalLucasKanade`

`DirectionalLucasKanade` gained `set_rigid_body_motion(rbm_ij)`: a per-frame
`(n_time_points, 2)` array giving a known, prescribed rigid-body translation.
The tracking window for every point now follows this prescribed motion, and
the motion is subtracted back out of the result, so `self.displacements`
reports the local motion relative to the rigid body motion rather than each
point's absolute pixel position. Only the component of the rigid body motion
aligned with each point's tracking direction (`dij`) is currently supported.
If `set_rigid_body_motion` is never called, it defaults to zero and existing
analyses are unaffected.

The same change also fixes the NumPy-path convergence check: `compute_delta`
(aliased as `compute_delta_numba`) returned a signed error, but the
optimizer's stopping test (`error < tol`) assumes a non-negative error, so
iterations could stop early on a spuriously negative error or fail to
converge. The error is now returned as its absolute value.

### Fixed

- **Asymmetric `pad` in `DirectionalLucasKanade` crashed every point.**
  `_interpolate_reference` and `_warm_up_kernels` paired the `(pad_y, pad_x)`
  axes in the opposite order to `_padded_slice`, so a non-square `pad` (e.g.
  `configure(pad=(2, 5))`) built the reference spline over a grid of the
  wrong shape. All three now use the same axis pairing.
- **A point already lost before a checkpoint could come back with garbage
  displacements after resuming.** `failed_points` is rebuilt from scratch on
  resume and is not itself checkpointed, so a resumed analysis had no record
  that a point was already `NaN`. `np.round(NaN).astype(int)` is undefined
  (e.g. `INT64_MIN` on x86, `0` on arm64) rather than raising, so such a point
  could silently restart tracking from a finite but meaningless position.
  `LucasKanade` and `DirectionalLucasKanade` now check the previous
  displacement for NaN/inf before rounding it and keep the point marked
  failed if so, matching what the compiled kernel already did.
- **A single untracked point (`NaN`) could break the displacement-vector
  display.** The napari `GUI` and the Qt `result_viewer` scaled vectors by
  `np.max`/`np.max(np.abs(...))`, both of which propagate to `NaN` if any
  point in the result failed to track. They now use `np.nanmax`, with a
  fallback when every point failed.
- With `processes` greater than one, warnings about failed points raised
  inside a worker used worker-local point indices and, under the
  `forkserver`/`spawn` start methods, might not reach the console at all. The
  parent process now re-summarises failed points with global indices once
  the worker results are merged, for both `LucasKanade` and
  `DirectionalLucasKanade`.
- **`compute_inverse_numba` and `compute_delta_numba` are importable from
  `LucasKanade` again.** The 1.4.0 numba rewrite renamed them to
  `compute_inverse` and `compute_delta`; code importing the 1.3.3 names broke
  as soon as it hit that import. Both old names are restored as aliases.

### Point selection consolidated on `SelectionGUI`

pyidi had accumulated five separate point-selection implementations. Three were
dead code, one was documented but no longer developed, and the one under active
development was not reachable from the documented workflow. There is now one.

- **`SubsetSelection` has been removed.** The tkinter widget in
  `pyidi/GUIs/selection.py` is gone, and `SelectionGUI` replaces it. The name is
  still importable, but instantiating it raises a `RuntimeError` naming the
  replacement, so existing scripts fail with an actionable message rather than an
  `ImportError`. Replace `SubsetSelection(video, roi_size=(21, 21), noverlap=0)`
  with `SelectionGUI(video, subset_size=21, subset_overlap=0)`.
- **`SelectionGUI` is now the documented interface.** It offers five selection
  methods (grid in a polygon, manual points, along a polyline, brush, and
  remove-point) plus automatic filtering by Shi-Tomasi corner strength or
  gradient direction. It requires the Qt extras: `pip install pyidi[qt]`.
- **Note for `LucasKanade` users:** `SubsetSelection` accepted an anisotropic
  `roi_size=(y, x)`; `SelectionGUI` currently takes a scalar `subset_size` and
  can only select square subsets. `LucasKanade.configure(roi_size=...)` still
  accepts a `(vertical, horizontal)` pair, so non-square ROIs remain available
  programmatically and through the napari `GUI`, just not through `SelectionGUI`.
- Removed the dead selection code: `tools.ManualROI`, `tools.GridOfROI` (both
  read a `video.reader.mraw` attribute that no longer exists), the unreachable
  `PickPoints` class in `_simplified_optical_flow.py`, and the stray
  `load_analysis copy.py`.

### Point validation

`set_points()` now validates its input instead of accepting almost anything.

- Empty input, non-2-D input, a wrong column count, and coordinates outside the
  image now raise `ValueError` with a message that says what was wrong. Empty and
  1-D input previously raised `IndexError: tuple index out of range`; out-of-range
  and negative coordinates were previously accepted silently, which since 1.4.0
  surfaced only as a `NaN` result much later.
- **Sub-pixel points are now rounded to the nearest pixel, with a warning.**
  Previously the same float input crashed in `SimplifiedOpticalFlow` (used
  directly as an array index) but was silently truncated *toward zero* in
  `LucasKanade`, `DirectionalLucasKanade`, and `DIC`. All four now agree, and
  round rather than truncate.
- `set_points()` accepts any object exposing a `.points` attribute, so a
  selection GUI instance can be passed directly. Previously only `SubsetSelection`
  was recognised, and passing the Qt GUI failed with an opaque error.
- The napari `GUI` now routes its selections through `set_points()` as well, so
  points picked in the UI get the same checks as programmatic ones.

### `SelectionGUI` editing

- **The last remaining grid or polyline can now be deleted.** Previously the
  delete button silently did nothing when only one entry was left, so you had to
  create a second grid just to remove the first. Deleting the last entry now
  works and re-seeds an empty one, which is what the old guard was clumsily
  protecting against.
- **Polygon and grid vertices can be dragged.** A left-drag starting within ~10
  screen pixels of an existing vertex moves it; a drag anywhere else still pans,
  and the grab radius is constant in screen pixels at any zoom. Clicking exactly
  on an existing vertex is now a no-op rather than adding a duplicate on top of
  it. The derived subset points are recomputed once when the drag finishes, not
  on every mouse-move.
- **Undo (Ctrl+Z)** for adding a vertex, moving a vertex, and deleting a grid or
  polyline. A restored grid comes back at its original row with its original
  label. Manual points, brush strokes and filter results are not undoable.
- The "Start new line" button now reads "Start new grid" in Grid mode. The
  status-bar hint said "Click 'Start new line' to begin a new grid" and now
  matches the button.

### Fixed

- **Mouse drags were offset from the cursor by 9 pixels.** The drag handlers read
  `ev.pos()`/`ev.buttonDownPos()`, which are local to the ViewBox, and passed them
  to `mapSceneToView()` and `sceneBoundingRect().contains()`, which expect scene
  coordinates. The click handlers already used `scenePos()` and were correct, so
  clicking and dragging disagreed. Most visibly this meant the **brush painted
  about 9 px away from the cursor**, and its bounds check was wrong by the same
  amount. All drag paths now use `scenePos()`/`buttonDownScenePos()`.

### Other

- New `pyidi/selection_geometry.py` holds the ROI-grid geometry as pure numpy,
  with no GUI-toolkit dependency, shared by the napari `GUI` and `SelectionGUI`.
  Its functions do not share one coordinate convention - each docstring states
  which one it uses, and the tests pin the difference deliberately.
- `SelectionGUI` accepts a numpy array as documented. A 2-D or 3-D array
  previously raised `AttributeError` because the frame was only set for a
  `VideoReader`; anything unusable now raises `TypeError`.
- First tests for the GUI package: `tests/test_selection_geometry.py` and
  `tests/test_set_points_validation.py` (24 tests).
- Fixed `README.md`, which told users to call `video.set_points(...)`.
  `VideoReader` has no such method - points are set on the method object.

## 1.4.0

### Lucas-Kanade performance

The inner optimization loop is now compiled with `numba` and parallelized over
points. Measured against 1.3.3 on the same machine, with identical results:

| case | 1.3.3 | 1.4.0 | speedup |
| --- | --- | --- | --- |
| `data_synthetic.cih`, 200 points, 101 frames | 7.94 s | 0.10 s | 77x |
| synthetic 512x512, 400 points, 150 frames | 21.04 s | 0.24 s | 89x |
| `data_synthetic.mp4`, 60 points, 10 frames | 2.16 s | 0.06 s | 36x |

The compiled path is on by default. `configure(use_compiled_kernel=False)`
selects the previous NumPy implementation, which is itself faster than 1.3.3
(about 1.6x, and more on video and image files) because the frame is no longer
re-read for every point. Results from the two paths agree to floating-point
round-off. If numba cannot be imported, pyidi falls back to the NumPy
implementation automatically and warns, rather than failing to import.

The compiled kernel supports cubic interpolation only; `int_order` other than `3`
falls back to the NumPy implementation and warns once.

### Directional Lucas-Kanade performance

`DirectionalLucasKanade` now uses the same compiled kernel machinery. The spline
evaluation is shared with Lucas-Kanade; only the least-squares solve differs,
because the motion is constrained to one prescribed direction and so has a
single unknown instead of two. Measured against 1.3.3 on the same machine, with
identical results:

| case | 1.3.3 | 1.4.0 | speedup |
| --- | --- | --- | --- |
| `data_synthetic.cih`, 120 points, 101 frames | 1.46 s | 0.04 s | 38x |
| synthetic 512x512, 400 points, 150 frames | 7.46 s | 0.11 s | 65x |
| `data_synthetic.mp4`, 60 points, 10 frames | 1.25 s | 0.03 s | 41x |

As for Lucas-Kanade, the frame is now read once per time step instead of once
per point. That alone makes the NumPy path about 18x faster on video and image
files; on memory-mapped formats it makes no measurable difference.

### Behaviour changes

- **A point that cannot be tracked no longer aborts the analysis.** Previously a
  degenerate region of interest raised `ValueError` and no results were returned
  at all. Such a point is now `NaN` from the frame at which it was lost, every
  other point is computed normally, and a warning is issued. Which points failed
  is recorded in the new `failed_points` attribute. This applies to
  `use_compiled_kernel=False` as well, so code that relied on the exception to
  halt should check for `NaN` instead. It applies to both `LucasKanade` and
  `DirectionalLucasKanade`.
- The NumPy path now also detects a runaway optimization. Previously such a point
  returned displacements of tens of pixels, silently.
- `DirectionalLucasKanade.configure()` no longer accepts `use_numba`. The
  argument was accepted but documented as not implemented and did nothing; it is
  replaced by `use_compiled_kernel`, which defaults to `True` and does what the
  old name suggested. Calls that passed `use_numba` need updating.
- `DirectionalLucasKanade` previously warned and continued with a zero update
  when the image had no gradient along the search direction, which returned a
  point that silently never moved. Such a point is now reported through
  `failed_points` and set to `NaN`.
- `DirectionalLucasKanade.configure()` accepts a scalar `pad` in addition to the
  `(pad_y, pad_x)` pair it required before.

Detection of untrackable points is best effort: a point can return implausible
values without being flagged, so a result without `NaN` is not proof that every
point tracked correctly.

### Other

- `numba` now requires at least 0.59.
- Windows and macOS added to the CI test matrix.
- When `processes` is greater than one, each worker is limited to a single numba
  thread, so process-level and thread-level parallelism cannot oversubscribe the
  CPU. This applies to both Lucas-Kanade methods.
- pyidi now asks numba for a fork-safe threading layer as the first thing it
  does, before importing anything that imports numba. This matters because
  pyMRAW starts numba's thread pool at import time, and the layer cannot be
  changed once the pool is up. Left alone it resolves to whatever is installed:
  TBB on machines that have it, but OpenMP on machines that do not, and GNU
  OpenMP kills any child forked from a process that has used it. Set
  `NUMBA_THREADING_LAYER` to override.
- On Linux, the worker pool no longer forks when a GNU OpenMP runtime is loaded
  into the process, since libgomp terminates any child forked from a process
  that has used it. numba's own threading layer is fork-safe, but OpenCV and
  some BLAS and SciPy builds pull in libgomp too, and that was enough to kill
  every worker. Such sessions now start their workers through `forkserver`.
  Where forking is safe it is still used, because it shares the video with the
  workers rather than pickling a copy to each of them.
- The kernels are compiled once in the parent process before the worker pool is
  created, instead of once per worker.
- `DirectionalLucasKanade`'s `compute_delta_numba` is now actually compiled; its
  `@numba.njit` decorator had been commented out. The name stays importable as an
  alias of the new `compute_delta`.
