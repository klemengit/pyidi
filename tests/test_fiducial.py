import numpy as np
import sys, os
from sdypy.io import sfmov
import cv2
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi

def test_instance():
    """
    Test creation of Fiducial instance with valid video data.
    Checks that the loaded video is a NumPy array with correct dimensions,
    and verifies internal conversion to grayscale if needed.
    Also tests that invalid input shapes raise the expected error.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    
    # Load the video (grayscale or RGB)
    data = sfmov.get_data(video_path)

    # Verify loaded data type and dimensions
    assert isinstance(data, np.ndarray), "Loaded data is not a NumPy array"
    assert data.ndim in [3, 4], f"Expected 3D or 4D array, got {data.ndim}D"

    # Instantiate Fiducial and check attributes
    test = pyidi.fiducial.Fiducial(data)
    assert isinstance(test, pyidi.fiducial.Fiducial), "Failed to create Fiducial instance"
    assert hasattr(test, 'video'), "Fiducial object missing 'video' attribute"

    # Check that internal video is grayscale with matching frame count
    assert test.video.ndim == 3, f"Expected grayscale 3D video, got {test.video.ndim}D"
    assert test.video.shape[0] == data.shape[0], "Frame count mismatch after conversion"
    
    # Test that invalid input shapes raise ValueError
    invalid_data = np.random.rand(64, 64)
    with pytest.raises(ValueError):
        pyidi.fiducial.Fiducial(invalid_data)


def test_instance_rgb2gray():
    """
    Test that an RGB video input is correctly converted to grayscale internally.
    """
    # Create dummy RGB video: 10 frames, 64x64 pixels, 3 color channels
    rgb_data = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    
    # Instantiate Fiducial and check internal video shape
    test = pyidi.fiducial.Fiducial(rgb_data)
    assert test.video.shape == (10, 64, 64), "RGB to grayscale conversion failed"


def test_compensation():
    """
    Test that the uncertainty (transformation error) at the reference frame after
    fiducial-based compensation is effectively zero.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    data = sfmov.get_data(video_path)
    test = pyidi.fiducial.Fiducial(data)
    
    # Pre-process video for better marker detection
    processed = test.pre_process(clip_range=(19.4, 20.0), enhance_contrast=True)
    
    # Detect fiducial markers in processed frames
    fiducials = test.detect_markers(processed)
    
    # Pick a reference frame from those with detected markers (deterministic for CI)
    valid_indices = [i for i, f in enumerate(fiducials) if f != 'none']
    ref = int(np.random.default_rng(0).choice(valid_indices))
    
    # Compute transformation matrices relative to the reference frame
    matrices = test.compute_transformations(fiducials, reference_index=ref)
    
    # Revert fiducial coordinates using the computed transformations
    fiducial_reverted = test.revert_fiducial(fiducials, matrices)
    
    # Analyze uncertainty (mean error) between original and reverted fiducials
    stats = test.uncertainty_analysis(fiducials, fiducial_reverted)

    # Assert that the error at the reference frame is negligible (within tolerance)
    assert abs(stats['Per-frame Mean Error'][ref]) < 1e-3


def test_revert_frames_shape():
    """
    Test that the video frames reverted (compensated) using transformation matrices
    have the same shape as the original video.
    """
    video_path = os.path.join(my_path, '..', 'data', 'data_fiducial.sfmov')
    data = sfmov.get_data(video_path)
    test = pyidi.fiducial.Fiducial(data)
    
    # Pre-process video and detect fiducials
    processed = test.pre_process(clip_range=(19.4, 20.0), enhance_contrast=True)
    fiducials = test.detect_markers(processed)
    
    # Compute transformations
    matrices = test.compute_transformations(fiducials)
    
    # Apply compensation to video frames
    compensated = test.revert_frames(matrices)
    
    # Verify that compensated video shape matches original video shape
    assert compensated.shape == test.video.shape, "Compensated video shape mismatch"



def _synthetic_aruco_video(dtype=np.uint8, scale=1):
    """A short recording of two ArUco markers translating by a known amount.

    Self-contained, so the checks below do not need the .sfmov data file.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    first = cv2.aruco.generateImageMarker(dictionary, 0, 60)
    second = cv2.aruco.generateImageMarker(dictionary, 1, 60)

    frames = []
    for dy, dx in [(0, 0), (5, 3), (11, 7), (-4, 9)]:
        frame = np.full((300, 300), 255, np.uint8)
        frame[40 + dy:100 + dy, 40 + dx:100 + dx] = first
        frame[180 + dy:240 + dy, 200 + dx:260 + dx] = second
        frames.append(frame)

    return np.stack(frames).astype(dtype) * scale


def test_detect_markers_without_pre_processing():
    """
    ``pre_process`` is documented as optional, so detection has to accept the
    array the class already holds. It used to demand a list and so raised
    ValueError on its own default.
    """
    test = pyidi.fiducial.Fiducial(_synthetic_aruco_video())

    from_attribute = test.detect_markers()
    from_list = test.detect_markers(list(test.video))

    assert len(from_attribute) == len(test.video), "One result per frame expected"
    assert all(frame for frame in from_attribute), "Markers should be found in every frame"
    assert from_attribute == from_list, "An array and a list of frames must agree"


def test_detect_markers_rejects_unusable_input():
    """Detection says what is wrong rather than failing inside OpenCV."""
    test = pyidi.fiducial.Fiducial(_synthetic_aruco_video())

    with pytest.raises(ValueError, match="8-bit"):
        pyidi.fiducial.Fiducial(_synthetic_aruco_video(np.uint16, 257)).detect_markers()

    with pytest.raises(ValueError):
        test.detect_markers("not a video")

    with pytest.raises(ValueError):
        test.detect_markers(np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError):
        test.detect_markers([])


def test_revert_frames_undoes_known_motion():
    """
    The markers move rigidly, so reverting has to put every frame back onto the
    reference one. The result is float, so that a frame which could not be
    reverted stays NaN instead of casting to a black one.
    """
    video = _synthetic_aruco_video()
    test = pyidi.fiducial.Fiducial(video)

    markers = test.detect_markers()
    matrices = test.compute_transformations(markers, transform_type="euclidean")
    reverted = test.revert_frames(matrices)

    assert reverted.shape == video.shape
    assert np.issubdtype(reverted.dtype, np.floating), "NaN must be representable"

    reference = video[0].astype(float)
    for i, frame in enumerate(reverted):
        covered = np.isfinite(frame)
        assert covered.any(), f"Frame {i} was not reverted at all"
        assert np.abs(frame[covered] - reference[covered]).max() == 0, (
            f"Frame {i} does not land back on the reference frame"
        )


def test_revert_frames_marks_a_skipped_frame_as_nan():
    """A frame with no transformation is NaN, not a black frame."""
    test = pyidi.fiducial.Fiducial(_synthetic_aruco_video())

    matrices = test.compute_transformations(test.detect_markers())
    matrices[2] = None

    reverted = test.revert_frames(matrices)
    assert np.isnan(reverted[2]).all(), "A skipped frame must be entirely NaN"
    assert np.isfinite(reverted[0]).any(), "Other frames are unaffected"

if __name__ == '__main__':
    test_instance()
    test_instance_rgb2gray()
    test_compensation()
    test_revert_frames_shape()
    test_detect_markers_without_pre_processing()
    test_detect_markers_rejects_unusable_input()
    test_revert_frames_undoes_known_motion()
    test_revert_frames_marks_a_skipped_frame_as_nan()
