# Changelog

This changelog starts at version 1.4.0. For earlier versions see the
[commit history](https://github.com/ladisk/pyidi/commits/master).

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
  CPU. The kernel also requests a fork-safe numba threading layer, since the
  OpenMP layer cannot survive `fork`. This applies to both Lucas-Kanade methods.
- The kernels are compiled once in the parent process before the worker pool is
  created, instead of once per worker.
- `DirectionalLucasKanade`'s `compute_delta_numba` is now actually compiled; its
  `@numba.njit` decorator had been commented out. The name stays importable as an
  alias of the new `compute_delta`.
