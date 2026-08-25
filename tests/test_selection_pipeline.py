"""
Tests for ``pyidi/selection/``, the headless mask -> evaluate -> select pipeline.

Nothing here imports Qt. The whole point of the package is that the pipeline is
usable without an interface, so these tests exercise it the way a script would.

The three steps are tested separately and then together:

- **evaluate** -- the score image, its NaN border, and agreement with a direct
  per-subset reference implementation. The vectorised form is the reason dense
  evaluation is affordable at all, so "does it compute the same number" is the
  load-bearing assertion in this file.
- **mask** -- what each entry kind rasterises to, how roles decide whether an
  entry contributes an area or coordinates, and how deselection is applied
  without destroying the geometry it came from.
- **select** -- threshold, minimum-distance suppression and the merge with
  hand-picked points, which are what stop a dense score image from returning a
  blob of adjacent pixels on every corner.
"""

import numpy as np
import pytest
from scipy.ndimage import sobel

from pyidi.selection import (
    DEFAULT_ROLE,
    Entry,
    ScoreStore,
    SelectionPipeline,
    all_literal_points,
    apply_deselection,
    as_point_array,
    available_evaluators,
    combined_mask,
    decimate,
    evaluate,
    get_evaluator,
    half_window,
    literal_points,
    merge_points,
    occupancy,
    rasterize,
    select,
    select_lattice,
    select_peaks,
    ROBUST_MAXIMUM_PERCENTILE,
    THRESHOLD_MODES,
    select_points,
    threshold_value,
    window_size,
)


# ---------------------------------------------------------------------------
# Fixtures -- one image with a flat region, a straight edge, a corner and a
# speckle patch, so a single frame can answer most of the questions below.
# ---------------------------------------------------------------------------

@pytest.fixture
def image():
    """160x200 uint16 frame: flat background, a bright square, a speckle patch."""
    img = np.full((160, 200), 40, dtype=np.uint16)
    img[30:80, 30:80] = 220          # square: corners at (30, 30) etc, edges between
    rng = np.random.default_rng(12345)
    img[100:150, 100:190] = rng.integers(20, 240, (50, 90))   # speckle
    return img


@pytest.fixture
def speckle():
    """A frame that is speckle everywhere, so every subset has something to score."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (128, 128)).astype(np.uint16)


def reference_shi_tomasi(image, row, col, half):
    """Score one subset the slow way, as ``SelectionGUI`` does it.

    Gradients are taken over the whole image rather than over the isolated ROI:
    that is the one intentional difference between the old implementation and
    the new one, and it is not what this reference is here to check.
    """
    img = np.asarray(image, dtype=np.float64)
    g_row, g_col = sobel(img, axis=0), sobel(img, axis=1)
    box = (slice(row - half[0], row + half[0] + 1), slice(col - half[1], col + half[1] + 1))
    a = float((g_col[box] ** 2).sum())
    c = float((g_row[box] ** 2).sum())
    b = float((g_col[box] * g_row[box]).sum())
    return float(np.linalg.eigvalsh(np.array([[a, b], [b, c]]))[0])


def rect(r0, c0, r1, c1):
    """A rectangular polygon as ``(row, col)`` vertices."""
    return [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]


# ---------------------------------------------------------------------------
# evaluate -- shape, dtype, window and border
# ---------------------------------------------------------------------------

def test_score_image_has_the_image_shape_and_is_float32(image):
    score = evaluate(image, 'shi_tomasi', 11)
    assert score.shape == image.shape
    assert score.dtype == np.float32


def test_window_is_odd_even_for_an_even_subset_size():
    assert window_size(11) == (11, 11)
    assert window_size(10) == (11, 11)      # 2 * (10 // 2) + 1
    assert window_size((21, 7)) == (21, 7)
    assert half_window((21, 7)) == (10, 3)


def test_border_is_nan_and_the_interior_is_finite(image):
    score = evaluate(image, 'shi_tomasi', 11)
    assert np.isnan(score[:5]).all()
    assert np.isnan(score[-5:]).all()
    assert np.isnan(score[:, :5]).all()
    assert np.isnan(score[:, -5:]).all()
    assert np.isfinite(score[5:-5, 5:-5]).all()


def test_anisotropic_border_depth_differs_per_axis(image):
    score = evaluate(image, 'shi_tomasi', (21, 7))
    assert np.isnan(score[:10]).all()
    assert np.isnan(score[:, :3]).all()
    assert np.isfinite(score[10:-10, 3:-3]).all()


def test_nan_never_passes_a_threshold(image):
    score = evaluate(image, 'shi_tomasi', 11)
    # The invalid border is excluded by the comparison itself, which is why no
    # separate validity test is threaded through the selectors.
    assert not (score[:5] > -np.inf).any()
    assert not (score > np.nanmin(score) - 1)[np.isnan(score)].any()


# ---------------------------------------------------------------------------
# evaluate -- the evaluators themselves
# ---------------------------------------------------------------------------

def test_shi_tomasi_matches_the_per_subset_reference(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    half = half_window(11)
    ours, reference = [], []
    for row in range(20, 110, 17):
        for col in range(20, 110, 13):
            ours.append(float(score[row, col]))
            reference.append(reference_shi_tomasi(speckle, row, col, half))
    ours, reference = np.array(ours), np.array(reference)
    np.testing.assert_allclose(ours, reference, rtol=1e-4, atol=1e-4 * reference.max())


def test_flat_image_scores_exactly_zero():
    score = evaluate(np.full((60, 60), 17, dtype=np.uint16), 'shi_tomasi', 11)
    finite = score[np.isfinite(score)]
    assert finite.size
    assert (finite == 0.0).all()


def test_a_corner_outscores_an_edge(image):
    score = evaluate(image, 'shi_tomasi', 11)
    corner = score[30, 30]          # top-left corner of the bright square
    edge = score[55, 30]            # midpoint of its left edge
    assert corner > edge
    assert edge >= 0.0


def test_gradient_direction_is_selective(speckle):
    stripes = np.zeros((80, 80), dtype=np.uint16)
    stripes[:, ::8] = 255           # intensity varies along columns only
    along_cols = evaluate(stripes, 'gradient_direction', 11, direction=(0, 1))
    along_rows = evaluate(stripes, 'gradient_direction', 11, direction=(1, 0))
    interior = (slice(10, -10), slice(10, -10))
    assert along_cols[interior].mean() > along_rows[interior].mean()


def test_gradient_direction_normalises_its_direction(speckle):
    one = evaluate(speckle, 'gradient_direction', 11, direction=(0, 1))
    five = evaluate(speckle, 'gradient_direction', 11, direction=(0, 5))
    np.testing.assert_allclose(one[10:-10, 10:-10], five[10:-10, 10:-10], rtol=1e-6)


def test_zero_direction_is_rejected(speckle):
    with pytest.raises(ValueError, match='non-zero'):
        evaluate(speckle, 'gradient_direction', 11, direction=(0, 0))


# ---------------------------------------------------------------------------
# evaluate -- registry
# ---------------------------------------------------------------------------

def test_registry_lists_the_built_in_evaluators():
    names = available_evaluators()
    assert 'shi_tomasi' in names
    assert 'gradient_direction' in names


def test_unknown_evaluator_names_the_registered_ones(speckle):
    with pytest.raises(ValueError, match='shi_tomasi'):
        evaluate(speckle, 'no_such_evaluator', 11)


def test_unknown_parameter_is_rejected(speckle):
    with pytest.raises(ValueError, match='direction'):
        evaluate(speckle, 'gradient_direction', 11, dirction=(0, 1))


def test_every_evaluator_parameter_is_described():
    spec = get_evaluator('gradient_direction')
    described = {p.name for p in spec.parameters}
    assert described == {'direction'}
    parameter = spec.parameters[0]
    assert parameter.kind == 'direction'
    assert parameter.default == (0.0, 1.0)


# ---------------------------------------------------------------------------
# evaluate -- bounding-box crop
# ---------------------------------------------------------------------------

def test_cropped_evaluation_matches_uncropped_inside_the_mask(image):
    mask = np.zeros(image.shape, dtype=bool)
    mask[40:70, 40:70] = True
    full = evaluate(image, 'shi_tomasi', 11, crop=False)
    cropped = evaluate(image, 'shi_tomasi', 11, mask=mask, crop=True)
    np.testing.assert_allclose(full[mask], cropped[mask], rtol=1e-5)


def test_cropped_evaluation_is_nan_outside_the_padded_box(image):
    mask = np.zeros(image.shape, dtype=bool)
    mask[40:70, 40:70] = True
    cropped = evaluate(image, 'shi_tomasi', 11, mask=mask, crop=True)
    assert np.isnan(cropped[:30]).all()
    assert np.isnan(cropped[85:]).all()


def test_an_empty_mask_evaluates_to_all_nan(image):
    empty = np.zeros(image.shape, dtype=bool)
    assert np.isnan(evaluate(image, 'shi_tomasi', 11, mask=empty, crop=True)).all()


def test_dense_evaluation_is_fast_enough_to_be_interactive():
    import time

    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (1000, 1000)).astype(np.uint16)
    evaluate(big, 'shi_tomasi', 11)          # warm scipy up
    start = time.perf_counter()
    evaluate(big, 'shi_tomasi', 11)
    elapsed = time.perf_counter() - start
    # A generous ceiling: the point is that this is not the minutes a per-subset
    # loop would take, not to pin a particular machine's timing.
    assert elapsed < 0.5, f'dense evaluation took {elapsed:.3f} s'


# ---------------------------------------------------------------------------
# scores -- caching
# ---------------------------------------------------------------------------

def test_a_repeated_request_is_served_from_cache(speckle):
    store = ScoreStore(speckle, 11)
    store.define('corners', 'shi_tomasi')
    first = store.get('corners')
    assert store.n_evaluations == 1
    second = store.get('corners')
    assert store.n_evaluations == 1
    assert first is second


def test_two_named_scores_coexist(speckle):
    store = ScoreStore(speckle, 11)
    store.define('corners', 'shi_tomasi')
    store.define('sideways', 'gradient_direction', direction=(0, 1))
    corners = store.get('corners')
    store.get('sideways')
    assert store.is_cached('corners')
    assert store.get('corners') is corners
    assert set(store.names) == {'corners', 'sideways'}


def test_the_same_computation_under_two_names_is_computed_once(speckle):
    store = ScoreStore(speckle, 11)
    store.define('a', 'shi_tomasi')
    store.define('b', 'shi_tomasi')
    store.get('a')
    store.get('b')
    assert store.n_evaluations == 1


def test_a_subset_size_change_invalidates_every_score(speckle):
    store = ScoreStore(speckle, 11)
    store.define('corners', 'shi_tomasi')
    store.get('corners')
    store.set_subset_size(21)
    assert not store.is_cached('corners')
    store.get('corners')
    assert store.n_evaluations == 2


def test_an_image_change_invalidates_every_score(speckle):
    store = ScoreStore(speckle, 11)
    store.define('corners', 'shi_tomasi')
    store.get('corners')
    store.set_image(speckle * 2)
    assert not store.is_cached('corners')


def test_list_and_tuple_parameters_share_a_cache_entry(speckle):
    store = ScoreStore(speckle, 11)
    store.define('a', 'gradient_direction', direction=[0, 1])
    store.define('b', 'gradient_direction', direction=(0, 1))
    store.get('a')
    store.get('b')
    assert store.n_evaluations == 1


# ---------------------------------------------------------------------------
# masks -- rasterisation and roles
# ---------------------------------------------------------------------------

def test_polygon_rasterises_to_its_interior():
    entry = Entry('polygon', rect(10, 20, 40, 60))
    mask = rasterize(entry, (80, 100))
    assert mask[25, 40]
    assert not mask[5, 40]
    assert not mask[25, 70]


def test_brush_geometry_is_used_as_the_mask():
    painted = np.zeros((50, 50), dtype=bool)
    painted[10:20, 10:20] = True
    entry = Entry('brush', painted)
    np.testing.assert_array_equal(rasterize(entry, (50, 50)), painted)


def test_a_brush_mask_of_the_wrong_shape_is_rejected():
    entry = Entry('brush', np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError, match='expected'):
        rasterize(entry, (50, 50))


def test_default_roles_follow_the_kind():
    assert Entry('polygon', rect(0, 0, 5, 5)).role == 'mask'
    assert Entry('brush', np.zeros((4, 4), bool)).role == 'mask'
    assert Entry('polyline', [(0, 0), (5, 5)]).role == 'points'
    assert Entry('points', [(1, 1)]).role == 'points'
    assert DEFAULT_ROLE['polygon'] == 'mask'


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match='Unknown entry kind'):
        Entry('rhombus', [])


def test_an_unknown_role_is_rejected():
    with pytest.raises(ValueError, match='role must be'):
        Entry('polygon', rect(0, 0, 5, 5), role='sometimes')


def test_the_combined_mask_is_the_union_of_visible_mask_entries():
    a = Entry('polygon', rect(10, 10, 30, 30))
    b = Entry('polygon', rect(20, 20, 40, 40))
    mask = combined_mask([a, b], (60, 60))
    assert mask[15, 15] and mask[35, 35] and mask[25, 25]


def test_a_hidden_entry_contributes_nothing():
    a = Entry('polygon', rect(10, 10, 30, 30))
    b = Entry('polygon', rect(40, 40, 55, 55), visible=False)
    mask = combined_mask([a, b], (60, 60))
    assert mask[15, 15]
    assert not mask[45, 45]


def test_a_points_role_entry_contributes_no_mask():
    entry = Entry('polygon', rect(10, 10, 30, 30), role='points')
    assert not combined_mask([entry], (60, 60)).any()


def test_a_mask_role_entry_contributes_no_literal_points():
    entry = Entry('points', [(5, 5)], role='mask')
    assert all_literal_points([entry], 11) == []


def test_changing_a_role_moves_the_contribution():
    entry = Entry('polygon', rect(10, 10, 40, 40))
    assert combined_mask([entry], (60, 60)).any()
    assert all_literal_points([entry], 11) == []

    entry.role = 'points'
    assert not combined_mask([entry], (60, 60)).any()
    assert len(all_literal_points([entry], 11)) > 0


def test_literal_points_honour_the_removed_set():
    entry = Entry('points', [(5, 5), (9, 9)])
    entry.removed.add((5, 5))
    assert literal_points(entry, 11) == [(9, 9)]


# ---------------------------------------------------------------------------
# masks -- deselection
# ---------------------------------------------------------------------------

def stroke_over(shape, r0, c0, r1, c1):
    """A rectangular deselect stroke."""
    painted = np.zeros(shape, dtype=bool)
    painted[r0:r1, c0:c1] = True
    return painted


def test_partial_deselection_keeps_the_remainder():
    shape = (60, 60)
    entry = Entry('polygon', rect(10, 10, 50, 50))
    apply_deselection([entry], stroke_over(shape, 10, 10, 30, 60), shape)
    mask = rasterize(entry, shape)
    assert not mask[20, 20]
    assert mask[40, 20]


def test_deselection_survives_a_subset_size_change():
    shape = (60, 60)
    entry = Entry('polygon', rect(10, 10, 50, 50))
    apply_deselection([entry], stroke_over(shape, 10, 10, 30, 60), shape)
    # The stroke was recorded on the entry, not baked into a derived point list,
    # so nothing about it depends on the subset size it was painted at.
    assert not rasterize(entry, shape)[20, 20]


def test_deselection_leaves_the_original_geometry_intact():
    shape = (60, 60)
    entry = Entry('polygon', rect(10, 10, 50, 50))
    original = list(entry.geometry)
    apply_deselection([entry], stroke_over(shape, 10, 10, 30, 60), shape)
    assert entry.geometry == original


def test_a_fully_covered_entry_is_reported_as_emptied():
    shape = (60, 60)
    entry = Entry('polygon', rect(10, 10, 50, 50))
    emptied = apply_deselection([entry], stroke_over(shape, 0, 0, 60, 60), shape)
    assert emptied == [entry]


def test_deselection_drops_covered_literal_points():
    shape = (60, 60)
    entry = Entry('points', [(15, 15), (45, 45)])
    apply_deselection([entry], stroke_over(shape, 10, 10, 20, 20), shape)
    assert entry.geometry == [(45, 45)]


# ---------------------------------------------------------------------------
# select -- threshold
# ---------------------------------------------------------------------------

def test_percentile_threshold_keeps_the_top_decile(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    limit = threshold_value(score, None, 'percentile', 90)
    points = select_peaks(score, separation=1, threshold=90,
                          threshold_mode='percentile', max_points=None)
    assert points
    assert all(score[r, c] > limit for r, c in points)


def test_the_fraction_of_the_maximum_rule_is_gone(speckle):
    """It was `quality` with a reference one dust mote could move."""
    score = evaluate(speckle, 'shi_tomasi', 11)
    assert 'fraction' not in THRESHOLD_MODES
    with pytest.raises(ValueError, match='fraction'):
        threshold_value(score, None, 'fraction', 0.5)


def test_an_unreachable_threshold_returns_an_empty_array(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    points = select(score, selector='peaks', threshold=100, threshold_mode='percentile')
    assert points.shape == (0, 2)
    assert points.dtype.kind == 'i'


def test_an_unknown_threshold_mode_is_rejected(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    with pytest.raises(ValueError, match='mode must be'):
        threshold_value(score, None, 'quantile', 0.9)


# ---------------------------------------------------------------------------
# select -- suppression
# ---------------------------------------------------------------------------

def pairwise_separation(points):
    """Smallest distance between any two points, or ``inf`` for fewer than two."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.inf
    diff = points[:, None, :] - points[None, :, :]
    distance = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(distance, np.inf)
    return distance.min()


def test_the_separation_is_respected(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    points = select_peaks(score, separation=10, threshold=50, threshold_mode='percentile')
    assert len(points) > 5
    assert pairwise_separation(points) >= 10


def test_a_separation_of_one_keeps_every_candidate(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    limit = threshold_value(score, None, 'percentile', 99)
    points = select_peaks(score, separation=1, threshold=99,
                          threshold_mode='percentile', max_points=None)
    assert len(points) == int((score > limit).sum())


def test_the_strongest_candidate_in_a_neighbourhood_wins():
    score = np.zeros((40, 40), dtype=np.float32)
    score[20, 20] = 5.0
    score[20, 23] = 9.0             # closer than the separation, and stronger
    points = select_peaks(score, separation=8, threshold=0, max_points=None)
    assert (20, 23) in points
    assert (20, 20) not in points


def test_a_dense_blob_yields_one_point():
    score = np.zeros((60, 60), dtype=np.float32)
    rows, cols = np.mgrid[0:60, 0:60]
    score += np.exp(-((rows - 30.0) ** 2 + (cols - 30.0) ** 2) / 40.0).astype(np.float32)
    points = select_peaks(score, separation=5, threshold=99,
                          threshold_mode='percentile', max_points=None)
    near = [p for p in points if abs(p[0] - 30) <= 5 and abs(p[1] - 30) <= 5]
    assert len(near) == 1


def test_the_point_cap_is_applied(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    many = select_peaks(score, separation=3, threshold=50,
                        threshold_mode='percentile', max_points=None)
    capped = select_peaks(score, separation=3, threshold=50,
                          threshold_mode='percentile', max_points=5)
    assert len(many) > 5
    assert len(capped) == 5
    assert capped == many[:5]


def test_results_are_deterministic(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    first = select(score, selector='peaks', separation=7, threshold=80, threshold_mode='percentile')
    second = select(score, selector='peaks', separation=7, threshold=80, threshold_mode='percentile')
    np.testing.assert_array_equal(first, second)


def test_ties_are_broken_by_position():
    score = np.zeros((40, 40), dtype=np.float32)
    score[10, 20] = 1.0
    score[12, 15] = 1.0             # equal score, larger row -- must lose
    points = select_peaks(score, separation=9, threshold=0.5,
                          threshold_mode='quality', max_points=None)
    assert points == [(10, 20)]


def test_selected_points_stay_inside_the_mask(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    mask = np.zeros(speckle.shape, dtype=bool)
    mask[30:90, 30:90] = True
    points = select_peaks(score, mask=mask, separation=6, threshold=50,
                          threshold_mode='percentile')
    assert points
    assert all(mask[r, c] for r, c in points)


def test_the_nan_border_is_never_selected(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    points = select_peaks(score, separation=4, threshold=10, threshold_mode='percentile')
    assert all(np.isfinite(score[r, c]) for r, c in points)


def test_selection_stays_interactive_on_a_dense_score_image():
    """A threshold drag re-selects on every step, so the whole thing is the budget.

    A loose threshold on a megapixel frame leaves ~10^5 pixels above it. Walking
    them all is 40 ms to 300 ms depending on the separation, which is why the
    candidates are reduced to the best in each cell first.
    """
    import time

    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (1000, 1000)).astype(np.uint16)
    score = evaluate(big, 'shi_tomasi', 11)
    assert (score > threshold_value(score, None, 'percentile', 90)).sum() > 50000

    start = time.perf_counter()
    points = select_peaks(score, separation=10, threshold=90, threshold_mode='percentile',
                          max_points=None)
    elapsed = time.perf_counter() - start
    assert len(points) > 100
    assert elapsed < 0.1, f'selection took {elapsed * 1000:.0f} ms'


def test_the_candidate_reduction_does_not_break_the_separation():
    """The cell grid is an approximation of the walk's input, never of its rule."""
    from pyidi.selection.select import CANDIDATE_CELL_FRACTION

    rng = np.random.default_rng(5)
    score = rng.random((300, 400)).astype(np.float32)
    for separation in (2, 4, 11, 30):
        points = np.array(select_peaks(score, separation=separation, threshold=0.5,
                                       threshold_mode='quality', max_points=None))
        assert len(points) > 10
        assert pairwise_separation(points) >= separation
        # ...and it really did reduce, wherever the cell is worth having
        if separation // CANDIDATE_CELL_FRACTION > 1:
            assert len(points) < (score > threshold_value(score, None, 'quality', 0.5)).sum()


def test_the_cell_grid_keeps_the_best_pixel_in_each_cell():
    from pyidi.selection.select import _block_best

    score = np.zeros((12, 12), dtype=np.float32)
    score[1, 1] = 1.0
    score[2, 2] = 5.0               # same 4x4 cell, stronger -- it should win
    score[9, 5] = 3.0
    rows, cols = _block_best(score, score > 0, 4)
    assert set(zip(rows.tolist(), cols.tolist())) == {(2, 2), (9, 5)}


def test_a_cell_with_nothing_eligible_contributes_nothing():
    """Unlike a lattice, which puts a point wherever the grid happens to fall."""
    from pyidi.selection.select import _block_best

    score = np.zeros((20, 20), dtype=np.float32)
    score[3, 3] = 1.0
    rows, _ = _block_best(score, score > 0, 5)
    assert len(rows) == 1


def nearest_neighbour(points):
    """Distance from each point to its closest other point."""
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    distance = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def test_keeping_every_nth_is_not_a_substitute_for_the_separation():
    """The measurement the separation control exists because of.

    Thinning the pixels above the threshold by keeping every n-th of them, in
    score order, is the obvious thing to reach for and it does not work:
    consecutive ranks are neighbours on the same feature, so most of what
    survives is still back-to-back with something else. On this noise field it
    leaves two fifths of the subsets within three pixels of another, and on a
    structured frame -- where the strong scores really are concentrated on a few
    features -- three quarters. The separation leaves none.
    """
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (300, 300)).astype(np.uint16)
    score = evaluate(image, 'shi_tomasi', 11)
    rows, cols = np.nonzero(score > threshold_value(score, None, 'quality', 0.05))
    order = np.argsort(-score[rows, cols], kind='stable')
    stride = max(2, rows.size // 2000)

    ranked = np.column_stack([rows[order][::stride], cols[order][::stride]])
    spaced = np.array(select_peaks(score, separation=6, threshold=0.05,
                                   threshold_mode='quality', max_points=None))

    assert (nearest_neighbour(ranked) < 3).mean() > 0.4
    assert (nearest_neighbour(spaced) < 3).mean() == 0
    assert pairwise_separation(spaced) >= 6


def test_a_zero_percentile_threshold_keeps_everything():
    """A slider at its loosest setting must not select nothing."""
    score = np.ones((40, 40), dtype=np.float32)
    assert threshold_value(score, None, 'percentile', 0) == -np.inf
    assert len(select_peaks(score, separation=1, threshold=0, max_points=None)) == 1600


# ---------------------------------------------------------------------------
# select -- lattice and decimation
# ---------------------------------------------------------------------------

def test_lattice_places_points_on_a_regular_grid():
    score = np.ones((60, 60), dtype=np.float32)
    points = select_lattice(score, pitch=12, threshold=0, max_points=None)
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    assert rows == [0, 12, 24, 36, 48]
    assert cols == [0, 12, 24, 36, 48]


def test_lattice_honours_the_threshold():
    score = np.ones((60, 60), dtype=np.float32)
    score[24, :] = 0.0
    points = select_lattice(score, pitch=12, threshold=0.5, threshold_mode='quality',
                            max_points=None)
    assert not any(r == 24 for r, _ in points)
    assert any(r == 12 for r, _ in points)


def test_unknown_selector_is_rejected(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    with pytest.raises(ValueError, match='Unknown selector'):
        select(score, selector='vibes')


def test_selector_ignores_parameters_it_does_not_take(speckle):
    """One set of defaults has to be usable with either selector."""
    score = evaluate(speckle, 'shi_tomasi', 11)
    points = select(score, selector='lattice', pitch=10, separation=99, threshold=0)
    assert len(points) > 0


def test_decimation_by_stride():
    points = [(i, i) for i in range(100)]
    assert decimate(points, stride=4) == points[::4]


def test_decimation_to_a_target_count():
    points = [(i, i) for i in range(100)]
    thinned = decimate(points, count=30)
    assert len(thinned) <= 30
    assert thinned[0] == points[0]
    assert thinned[-1] == points[-1]


def test_decimation_leaves_a_short_list_alone():
    points = [(1, 1), (2, 2)]
    assert decimate(points, count=10) == points


# ---------------------------------------------------------------------------
# select -- merging with hand-picked points
# ---------------------------------------------------------------------------

def test_literal_points_survive_a_low_score():
    score = np.zeros((60, 60), dtype=np.float32)
    score[40, 40] = 10.0
    literal = [(10, 10)]
    picked = select_peaks(score, separation=5, threshold=99, threshold_mode='percentile', max_points=None,
                          occupied=occupancy(literal, score.shape, 5))
    merged = merge_points(literal, picked)
    assert (merged == np.array([10, 10])).all(axis=1).any()


def test_selection_never_crowds_a_literal_point():
    score = np.ones((60, 60), dtype=np.float32)
    literal = [(30, 30)]
    picked = select_peaks(score, separation=8, threshold=0, max_points=None,
                          occupied=occupancy(literal, score.shape, 8))
    assert all(np.hypot(r - 30, c - 30) >= 8 for r, c in picked)


def test_a_coincident_point_appears_once():
    merged = merge_points([(5, 5)], [(5, 5), (9, 9)])
    assert merged.shape == (2, 2)


def test_as_point_array_of_nothing_is_shaped_for_indexing():
    empty = as_point_array([])
    assert empty.shape == (0, 2)
    assert empty[:, 0].size == 0


# ---------------------------------------------------------------------------
# pipeline -- end to end
# ---------------------------------------------------------------------------

def test_end_to_end_returns_points_inside_the_polygon(image):
    entry = Entry('polygon', rect(100, 100, 150, 190))
    points = select_points(image, [entry], subset_size=11, separation=8)
    assert len(points) > 5
    assert points.dtype.kind == 'i'
    assert points.shape[1] == 2
    assert ((points[:, 0] >= 100) & (points[:, 0] <= 150)).all()
    assert ((points[:, 1] >= 100) & (points[:, 1] <= 190)).all()


def test_no_mask_entries_gives_no_points(image):
    assert select_points(image, []).shape == (0, 2)


def test_a_points_role_entry_needs_no_score(image):
    entry = Entry('points', [(50, 50), (60, 60)])
    points = select_points(image, [entry])
    np.testing.assert_array_equal(points, np.array([[50, 50], [60, 60]]))


def test_mask_edits_do_not_re_evaluate(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    pipeline.get_points()
    evaluations = pipeline.store.n_evaluations
    assert evaluations == 1

    entry.geometry = rect(105, 105, 145, 185)
    pipeline.selector_params.update({'threshold': 95, 'threshold_mode': 'percentile'})
    pipeline.get_points()
    assert pipeline.store.n_evaluations == evaluations


def test_a_subset_size_change_does_re_evaluate(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    pipeline.get_points()
    pipeline.set_subset_size(21)
    pipeline.get_points()
    assert pipeline.store.n_evaluations == 2


def test_hiding_and_unhiding_restores_the_points_without_re_evaluating(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    before = pipeline.get_points()
    evaluations = pipeline.store.n_evaluations

    entry.visible = False
    assert pipeline.get_points().shape == (0, 2)

    entry.visible = True
    np.testing.assert_array_equal(pipeline.get_points(), before)
    assert pipeline.store.n_evaluations == evaluations


def test_per_entry_settings_are_honoured(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    loose = pipeline.add_entry('polygon', rect(100, 100, 125, 190))
    tight = pipeline.add_entry('polygon', rect(126, 100, 150, 190))
    loose.selector_params = {'separation': 4, 'threshold': 50, 'threshold_mode': 'percentile'}
    tight.selector_params = {'separation': 20, 'threshold': 50, 'threshold_mode': 'percentile'}

    points = pipeline.get_points()
    in_loose = points[points[:, 0] <= 125]
    in_tight = points[points[:, 0] >= 126]
    assert len(in_loose) > len(in_tight)
    assert pairwise_separation(in_tight) >= 20


def test_uniform_per_entry_settings_equal_global_settings(image):
    shared = {'separation': 9, 'threshold': 70, 'threshold_mode': 'percentile'}

    globally = SelectionPipeline(image, subset_size=11)
    globally.add_entry('polygon', rect(100, 100, 150, 190))
    globally.selector_params.update(shared)

    per_entry = SelectionPipeline(image, subset_size=11)
    entry = per_entry.add_entry('polygon', rect(100, 100, 150, 190))
    entry.selector_params = dict(shared)

    np.testing.assert_array_equal(globally.get_points(), per_entry.get_points())


def test_entries_sharing_settings_compete_for_the_same_separation(image):
    """Two adjacent regions filtered alike must not place points on their shared edge."""
    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', rect(100, 100, 124, 190))
    pipeline.add_entry('polygon', rect(125, 100, 150, 190))
    pipeline.selector_params.update({'separation': 12, 'threshold': 40,
                                    'threshold_mode': 'percentile'})
    points = pipeline.get_points()
    assert pairwise_separation(points) >= 12


def test_literal_and_selected_points_combine(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    pipeline.add_entry('points', [(60, 60)])
    pipeline.selector_params.update({'separation': 8, 'threshold': 60,
                                    'threshold_mode': 'percentile'})
    points = pipeline.get_points()
    assert (points == np.array([60, 60])).all(axis=1).any()
    assert len(points) > 1


def test_labels_are_never_reused(image):
    pipeline = SelectionPipeline(image)
    first = pipeline.add_entry('polygon', rect(10, 10, 20, 20))
    second = pipeline.add_entry('polygon', rect(30, 30, 40, 40))
    assert (first.label, second.label) == ('Polygon 1', 'Polygon 2')
    pipeline.remove_entry(second)
    third = pipeline.add_entry('polygon', rect(50, 50, 60, 60))
    assert third.label == 'Polygon 3'


def test_deselecting_through_the_pipeline_drops_emptied_entries(image):
    pipeline = SelectionPipeline(image)
    entry = pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    stroke = np.ones(image.shape, dtype=bool)
    emptied = pipeline.deselect(stroke)
    assert emptied == [entry]
    assert pipeline.entries == []


def test_points_property_matches_get_points(image):
    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('polygon', rect(100, 100, 150, 190))
    np.testing.assert_array_equal(pipeline.points, pipeline.get_points())


def test_output_is_accepted_by_a_method_class(image, tmp_path):
    import warnings

    from pyidi import SimplifiedOpticalFlow, VideoReader

    entry = Entry('polygon', rect(100, 100, 150, 190))
    points = select_points(image, [entry], subset_size=11, separation=10)
    assert len(points)

    video = VideoReader(np.stack([image, image]).astype(np.uint16), root=str(tmp_path))
    method = SimplifiedOpticalFlow(video)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        method.set_points(points)
    np.testing.assert_array_equal(method.points, points)


# ---------------------------------------------------------------------------
# select -- the quality threshold
#
# The rule the interface defaults to, and the reason it does. A percentile
# ranks *pixels*, and on a dense score image the pixels are overwhelmingly
# background, so most of a percentile slider's travel is spent inside the
# featureless area. Quality is measured against the best feature instead.
# ---------------------------------------------------------------------------

def flat_with_corners():
    """A frame like a real one: mostly blank, a few strong features, sensor noise."""
    img = np.full((200, 300), 240, dtype=np.uint8)
    corners = np.zeros(img.shape, dtype=bool)
    for row in range(40, 180, 60):
        for col in range(40, 280, 60):
            img[row:row + 20, col:col + 20] = 20
            corners[row - 9:row + 29, col - 9:col + 29] = True
    noise = np.random.default_rng(3).integers(-5, 6, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8), corners


def test_quality_is_a_fraction_of_the_robust_maximum(speckle):
    score = evaluate(speckle, 'shi_tomasi', 11)
    robust = np.nanpercentile(score, ROBUST_MAXIMUM_PERCENTILE)
    assert threshold_value(score, None, 'quality', 0.25) == pytest.approx(0.25 * robust)


def test_a_lone_outlier_barely_moves_the_quality_scale():
    """A specular highlight must not drag every useful setting into the slider's floor.

    This is the whole difference between `quality` and taking a fraction of the
    literal maximum, which is why the latter is not offered: on this score image
    it moves by a factor of fifty where quality moves by 2%.
    """
    score = np.random.default_rng(0).random((200, 300))
    before = threshold_value(score, None, 'quality', 0.1)
    literal_before = 0.1 * score.max()

    score[100, 150] = 500.0                     # one absurdly bright pixel
    assert threshold_value(score, None, 'quality', 0.1) == pytest.approx(before, rel=0.02)
    assert 0.1 * score.max() > 50 * literal_before


def test_a_zero_quality_keeps_everything():
    score = np.zeros((40, 40))
    assert threshold_value(score, None, 'quality', 0) == -np.inf


def test_quality_keeps_the_points_off_the_blank_background():
    """The headline behaviour: the whole slider stays inside the useful range."""
    image, corners = flat_with_corners()
    score = evaluate(image, 'shi_tomasi', 11)

    for quality in (0.5, 0.1, 0.01):
        points = np.array(select_peaks(score, separation=8, threshold=quality,
                                       threshold_mode='quality', max_points=None))
        assert len(points)
        assert corners[points[:, 0], points[:, 1]].all(), quality


def test_a_percentile_threshold_does_not():
    """Why the default changed: the same frame, ranked by pixel instead."""
    image, corners = flat_with_corners()
    score = evaluate(image, 'shi_tomasi', 11)

    points = np.array(select_peaks(score, separation=8, threshold=50,
                                   threshold_mode='percentile', max_points=None))
    assert corners[points[:, 0], points[:, 1]].mean() < 0.9


# ---------------------------------------------------------------------------
# select -- candidate ordering and suppression, at speed
# ---------------------------------------------------------------------------

def test_partitioning_the_candidates_matches_the_full_sort():
    """The `keep` shortcut must be an optimisation, not an approximation."""
    from pyidi.selection.select import _ordered_candidates

    rng = np.random.default_rng(5)
    for score in (rng.random((120, 160)), rng.integers(0, 4, (120, 160)).astype(float)):
        eligible = score > np.percentile(score, 20)
        full_rows, full_cols = _ordered_candidates(score, eligible)
        for keep in (1, 9, 400, 5000):
            rows, cols = _ordered_candidates(score, eligible, keep)
            np.testing.assert_array_equal(rows[:keep], full_rows[:keep])
            np.testing.assert_array_equal(cols[:keep], full_cols[:keep])


def test_chunked_suppression_matches_a_plain_walk():
    """Batching the occupancy test is exact because occupancy only ever grows."""
    from pyidi.selection.select import _disc, _ordered_candidates, suppress

    shape = (90, 130)
    score = np.random.default_rng(6).random(shape)
    rows, cols = _ordered_candidates(score, score > np.percentile(score, 30))

    for radius in (0, 1, 4, 11):
        for max_points in (None, 5, 10000):
            taken = np.zeros(shape, dtype=bool)
            expected = []
            for row, col in zip(rows.tolist(), cols.tolist()):
                if taken[row, col]:
                    continue
                expected.append((row, col))
                if max_points is not None and len(expected) >= max_points:
                    break
                radius = int(radius)
                if radius == 0:
                    taken[row, col] = True
                    continue
                disc = _disc(radius)
                r0, r1 = max(0, row - radius), min(shape[0], row + radius + 1)
                c0, c1 = max(0, col - radius), min(shape[1], col + radius + 1)
                taken[r0:r1, c0:c1] |= disc[r0 - row + radius:r1 - row + radius,
                                            c0 - col + radius:c1 - col + radius]
            assert suppress(rows, cols, shape, radius, max_points) == expected


# ---------------------------------------------------------------------------
# select -- decimation
# ---------------------------------------------------------------------------

def test_decimation_thins_the_points_without_moving_them(speckle):
    """The distinction from a wider minimum distance, which re-selects instead."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('brush', np.ones(speckle.shape, dtype=bool))
    pipeline.selector_params.update({'separation': 6, 'threshold': 0.05})

    base = {tuple(point) for point in pipeline.points.tolist()}
    assert len(base) > 20

    for stride in (2, 3, 7):
        pipeline.selector_params['decimation'] = stride
        thinned = [tuple(point) for point in pipeline.points.tolist()]
        assert set(thinned) <= base, stride
        assert len(thinned) == pytest.approx(len(base) / stride, rel=0.15)


def test_a_wider_separation_moves_the_points_instead(speckle):
    """The contrast decimation exists for."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('brush', np.ones(speckle.shape, dtype=bool))
    pipeline.selector_params.update({'separation': 6, 'threshold': 0.05})
    base = {tuple(point) for point in pipeline.points.tolist()}

    pipeline.selector_params['separation'] = 18
    respaced = {tuple(point) for point in pipeline.points.tolist()}
    assert not respaced <= base


def test_decimation_leaves_hand_placed_points_alone():
    """They were placed deliberately; thinning is for what the selector found."""
    image, _ = flat_with_corners()
    pipeline = SelectionPipeline(image, subset_size=11)
    pipeline.add_entry('brush', np.ones(image.shape, dtype=bool))
    pipeline.add_entry('points', [(100, 150), (100, 170), (100, 190)])
    pipeline.selector_params['decimation'] = 5

    points = {tuple(point) for point in pipeline.points.tolist()}
    assert {(100, 150), (100, 170), (100, 190)} <= points


def test_decimating_one_group_does_not_let_another_fill_the_gaps(speckle):
    """What was selected is stamped before it is thinned, so gaps stay gaps."""
    def right_hand_points(decimation):
        pipeline = SelectionPipeline(speckle, subset_size=11)
        left = pipeline.add_entry('polygon', rect(10, 10, 90, 58))
        right = pipeline.add_entry('polygon', rect(10, 60, 90, 110))
        # Different minimum distances, so the two are always separate groups:
        # entries that share their settings are deliberately selected together.
        left.selector_params = {'separation': 5, 'threshold': 0.05, 'decimation': decimation}
        right.selector_params = {'separation': 6, 'threshold': 0.05}
        return pipeline.points_and_credits()[1][1]

    undecimated = right_hand_points(1)
    assert len(undecimated) > 5
    np.testing.assert_array_equal(right_hand_points(4), undecimated)


# ---------------------------------------------------------------------------
# One pass for points and per-entry credits
# ---------------------------------------------------------------------------

def test_points_and_credits_agree_with_asking_separately(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('polygon', rect(10, 10, 90, 110))
    pipeline.add_entry('points', [(50, 50)])

    points, credited = pipeline.points_and_credits()
    np.testing.assert_array_equal(points, pipeline.get_points())
    for mine, theirs in zip(credited, pipeline.points_by_entry()):
        np.testing.assert_array_equal(mine, theirs)


# ---------------------------------------------------------------------------
# Candidates -- what the mask is leaving out
# ---------------------------------------------------------------------------

def test_candidates_ignore_the_mask_entirely(speckle):
    """They answer "what is there", which is the question a mask cannot."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('polygon', rect(5, 5, 25, 25))
    candidates = pipeline.candidate_points()
    assert len(candidates) > len(pipeline.points)
    assert not pipeline.mask[candidates[:, 0], candidates[:, 1]].all()


def test_candidates_survive_a_mask_edit(speckle):
    """Cached across mask changes, so painting a region re-selects nothing."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    before = pipeline.candidate_points()
    pipeline.add_entry('polygon', rect(5, 5, 25, 25))
    assert pipeline.candidate_points() is before


def test_candidates_are_recomputed_when_the_score_changes(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.define_score('score', 'shi_tomasi')
    before = pipeline.candidate_points()
    pipeline.define_score('score', 'gradient_direction', direction=(0, 1))
    assert pipeline.candidate_points() is not before


def test_candidates_are_recomputed_at_a_new_subset_size(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    before = pipeline.candidate_points()
    pipeline.set_subset_size(21)
    assert pipeline.candidate_points() is not before


def test_candidates_are_recomputed_when_a_selector_setting_changes(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    before = pipeline.candidate_points()
    pipeline.selector_params['separation'] = 3
    after = pipeline.candidate_points()
    assert after is not before
    assert len(after) > len(before)


# ---------------------------------------------------------------------------
# Removing a single point
#
# A selected point is not stored anywhere: it is re-derived from the score
# every time the pipeline runs. So "remove this one" cannot be a deletion --
# it has to be an edit to the mask that the next selection will respect.
# ---------------------------------------------------------------------------

def test_removing_a_point_leaves_no_replacement_beside_it(speckle):
    """Erasing the pixel alone is not enough.

    The reduction picks the best pixel of each block, so taking the winner away
    promotes its neighbour and the point comes back one or two pixels along --
    which reads as the click having nudged the point rather than removed it.
    What is erased is the whole disc the point was reserving, so nothing can
    land nearer to it than a neighbouring point legitimately could have.
    """
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(10, 10, 118, 118))
    separation = pipeline.selector_params['separation']

    target = tuple(int(v) for v in pipeline.points[0])
    pipeline.remove_point(entry, target)

    survivors = pipeline.points
    assert not any(tuple(p) == target for p in survivors)
    gaps = np.hypot(survivors[:, 0] - target[0], survivors[:, 1] - target[1])
    assert gaps.min() >= separation


def test_removing_a_point_works_more_than_once(speckle):
    """The regression: the second click and every one after it did nothing.

    ``erased`` was grown with an in-place write, and the rasterisation cache
    identifies that array by object -- so after the first click, which allocates
    it, no later one changed anything the cache could see.
    """
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(10, 10, 118, 118))

    for _ in range(5):
        target = tuple(int(v) for v in pipeline.points[0])
        pipeline.remove_point(entry, target)
        assert not any(tuple(p) == target for p in pipeline.points)


def test_removing_a_hand_picked_point_deletes_it(speckle):
    """Outright, so that clicking the same pixel again puts one back."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('points', [(30, 30), (60, 60)])
    pipeline.remove_point(entry, (30, 30))
    assert entry.geometry == [(60, 60)]
    assert entry.erased is None


def test_removing_a_point_from_a_polyline_records_it(speckle):
    """A polyline re-derives its points too, so the coordinate has to be kept."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polyline', [(20, 20), (20, 110)])
    target = literal_points(entry, pipeline.subset_size)[1]
    pipeline.remove_point(entry, target)
    assert target in entry.removed
    assert target not in literal_points(entry, pipeline.subset_size)


def test_removing_a_point_replaces_the_erased_array(speckle):
    """Never a write into it: see the contract on ``Entry.erased``."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(10, 10, 118, 118))
    pipeline.remove_point(entry, tuple(int(v) for v in pipeline.points[0]))
    first = entry.erased
    pipeline.remove_point(entry, tuple(int(v) for v in pipeline.points[0]))
    assert entry.erased is not first


# ---------------------------------------------------------------------------
# What a deselect stroke costs
#
# An `erased` array covers the frame; a stroke covers a few hundred pixels of
# it. Handing one to every region on the list would make a single dab cost a
# megabyte per region -- and cost it again in every undo snapshot.
# ---------------------------------------------------------------------------

def test_a_stroke_only_reaches_the_entries_it_covers(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    covered = pipeline.add_entry('polygon', rect(10, 10, 50, 50))
    missed = pipeline.add_entry('polygon', rect(80, 80, 120, 120))

    stroke = np.zeros(pipeline.shape, dtype=bool)
    stroke[20:25, 20:25] = True
    pipeline.deselect(stroke)

    assert covered.erased is not None
    assert missed.erased is None


def test_a_stroke_that_misses_everything_changes_nothing(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(10, 10, 50, 50))
    before = pipeline.points

    stroke = np.zeros(pipeline.shape, dtype=bool)
    stroke[100:105, 100:105] = True
    assert pipeline.deselect(stroke) == []

    assert entry.erased is None
    np.testing.assert_array_equal(pipeline.points, before)


def test_deselection_still_erases_what_it_does_cover(speckle):
    """The skip above must not have cost the stroke its job."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('polygon', rect(10, 10, 118, 118))
    before = pipeline.mask.sum()

    stroke = np.zeros(pipeline.shape, dtype=bool)
    stroke[40:60, 40:60] = True
    pipeline.deselect(stroke)

    assert pipeline.mask.sum() == before - stroke.sum()


# ---------------------------------------------------------------------------
# Points that are not on the image
#
# A click lands wherever the interface lets it land, and the view is always
# larger than the frame. A subset centred off the frame is not something that
# can be tracked, and it is an index error waiting for whichever array reads it
# first.
# ---------------------------------------------------------------------------

def test_a_hand_picked_point_off_the_frame_is_dropped(speckle):
    pipeline = SelectionPipeline(speckle, subset_size=11)
    pipeline.add_entry('points', [(-5, 40), (40, 40), (40, 999)])
    assert pipeline.literal_points() == [(40, 40)]
    assert [tuple(p) for p in pipeline.points] == [(40, 40)]


def test_an_off_frame_point_is_dropped_from_its_row_too(speckle):
    """Not just from the total, or the list would disagree with the canvas."""
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('points', [(-5, 40), (40, 40)])
    credited = pipeline.points_by_entry()[pipeline.entries.index(entry)]
    assert [tuple(p) for p in credited] == [(40, 40)]


def test_merging_survives_a_coordinate_off_the_top_of_the_frame():
    """The dedup folds a pair into one integer; the fold has to stay one-to-one."""
    merged = merge_points([(-5, 3), (-5, 3), (2, 7)], [(2, 7), (9, 1)])
    assert [tuple(p) for p in merged] == [(-5, 3), (2, 7), (9, 1)]


# ---------------------------------------------------------------------------
# The score cache is bounded
#
# Every distinct set of evaluator parameters is a separate full-frame float32.
# A spin box dragged through sixty values asks for sixty of them.
# ---------------------------------------------------------------------------

def test_the_score_cache_stops_growing(speckle):
    store = ScoreStore(speckle, subset_size=11, max_cached=4)
    for k in range(20):
        store.define('score', 'gradient_direction', direction=(1.0, k + 1.0))
        store.get('score')
    assert len(store._cache) == 4
    assert store.n_evaluations == 20


def test_the_cache_drops_the_least_recently_used(speckle):
    store = ScoreStore(speckle, subset_size=11, max_cached=2)
    for name, direction in (('a', (0.0, 1.0)), ('b', (1.0, 0.0))):
        store.define(name, 'gradient_direction', direction=direction)
        store.get(name)

    store.get('a')                       # 'b' is now the older of the two
    store.define('c', 'gradient_direction', direction=(1.0, 1.0))
    store.get('c')

    assert store.is_cached('a')
    assert store.is_cached('c')
    assert not store.is_cached('b')


def test_a_repeated_request_is_still_free(speckle):
    """The eviction must not have cost the cache its reason for existing."""
    store = ScoreStore(speckle, subset_size=11, max_cached=4)
    store.define('score', 'shi_tomasi')
    first = store.get('score')
    assert store.get('score') is first
    assert store.n_evaluations == 1


# ---------------------------------------------------------------------------
# The point cap, when something is already taken
# ---------------------------------------------------------------------------

def test_the_cap_is_filled_even_though_some_candidates_are_taken(speckle):
    """At a separation of 1 the candidates are only sorted as far as the cap.

    That is exact when every one of them is accepted, and it was not: the
    occupied positions were dropped *after* the cut, so a run with hand-picked
    points returned fewer points than the cap allowed while more were eligible.
    """
    score = evaluate(speckle, 'shi_tomasi', 11)
    mask = np.zeros(score.shape, dtype=bool)
    mask[20:110, 20:110] = True

    free = select_peaks(score, mask, separation=1, threshold=0, max_points=50)
    taken = occupancy(free[:10], score.shape, 0)
    blocked = select_peaks(score, mask, separation=1, threshold=0, max_points=50,
                           occupied=taken)

    assert len(blocked) == 50
    assert not set(blocked) & set(free[:10])


# ---------------------------------------------------------------------------
# The rasterisation cache is keyed by identity
# ---------------------------------------------------------------------------

def test_the_raster_cache_holds_the_entry_it_describes(speckle):
    """An ``id`` is unique only among live objects.

    Without a reference here, a deleted entry could be collected and a new one
    allocated at the same address, which would then read the dead entry's area
    out of the cache.
    """
    pipeline = SelectionPipeline(speckle, subset_size=11)
    entry = pipeline.add_entry('polygon', rect(10, 10, 50, 50))
    pipeline.area(entry)
    assert any(held is entry for held, _, _ in pipeline._raster_cache.values())
