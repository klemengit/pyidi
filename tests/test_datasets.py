import hashlib
import io
import os
import sys

import numpy as np
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyidi
from pyidi import datasets


HEADER = """<?xml version="1.0" encoding="utf-8"?>
<cih>
    <frameInfo>
        <recordedFrame>3000</recordedFrame>
        <totalFrame>3000</totalFrame>
        <startFrame>0</startFrame>
        <skipFrame>1</skipFrame>
        <playbackEndFrameNo>2999</playbackEndFrameNo>
        <playbackStartFrameNo>0</playbackStartFrameNo>
        <saveEndFrameNo>2999</saveEndFrameNo>
        <saveStartFrameNo>0</saveStartFrameNo>
    </frameInfo>
    <basicInfo>
        <skipFrame>1</skipFrame>
        <comment>Excerpt of music_box.cihx</comment>
    </basicInfo>
</cih>
"""


def test_data_dir(tmp_path, monkeypatch):
    assert datasets.get_data_dir(str(tmp_path / 'given')) == str(tmp_path / 'given')
    assert os.path.isdir(tmp_path / 'given')

    monkeypatch.setenv('PYIDI_DATA_DIR', str(tmp_path / 'from_env'))
    assert datasets.get_data_dir() == str(tmp_path / 'from_env')
    assert os.path.isdir(tmp_path / 'from_env')


def test_patch_header():
    patched = datasets._patch_header(HEADER, datasets.MUSIC_BOX, first_frame=100, n_frames=10)

    for tag, value in (('recordedFrame', 10), ('totalFrame', 10), ('startFrame', 0),
                       ('playbackStartFrameNo', 0), ('playbackEndFrameNo', 9),
                       ('saveStartFrameNo', 0), ('saveEndFrameNo', 9)):
        assert f'<{tag}>{value}</{tag}>' in patched

    # the skipFrame outside of <frameInfo> must not be touched
    assert patched.count('<skipFrame>1</skipFrame>') == 2
    assert 'Frames 100 to 109' in patched


def test_patch_header_without_frame_info():
    with pytest.raises(ValueError):
        datasets._patch_header('<cih></cih>', datasets.MUSIC_BOX, 0, 10)


def test_frame_range_is_validated(tmp_path):
    with pytest.raises(ValueError):
        datasets.fetch_music_box(n_frames=1, first_frame=-1, data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        datasets.fetch_music_box(n_frames=0, data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        datasets.fetch_music_box(n_frames=10, first_frame=2995, data_dir=str(tmp_path))


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, data, status):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Length': str(len(data))}

    def __exit__(self, *args):
        self.close()


def _fake_urlopen(content, status=206):
    """``urlopen`` replacement that honours the ``Range`` header of the request."""
    def urlopen(request, *args, **kwargs):
        header = request.get_header('Range')
        if header is None:
            return _FakeResponse(content, 200)
        first, last = header.replace('bytes=', '').split('-')
        stop = int(last) + 1 if last else len(content)
        return _FakeResponse(content[int(first):stop], status)
    return urlopen


def test_download_range(tmp_path, monkeypatch):
    content = bytes(range(256)) * 4
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', _fake_urlopen(content))

    destination = str(tmp_path / 'part.bin')
    datasets._download('https://example.org/f', destination, start=256, end=511, progress=False)

    assert open(destination, 'rb').read() == content[256:512]
    assert not os.path.exists(destination + '.part')


def test_download_resumes(tmp_path, monkeypatch):
    content = bytes(range(256)) * 4
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', _fake_urlopen(content))

    destination = str(tmp_path / 'part.bin')
    with open(destination + '.part', 'wb') as f:      # an interrupted download
        f.write(content[256:300])
    datasets._download('https://example.org/f', destination, start=256, end=511, progress=False)

    assert open(destination, 'rb').read() == content[256:512]


def test_download_without_range_support(tmp_path, monkeypatch):
    content = bytes(range(256)) * 4
    monkeypatch.setattr(datasets.urllib.request, 'urlopen', _fake_urlopen(content, status=200))

    destination = str(tmp_path / 'part.bin')
    with pytest.raises(ConnectionError):
        datasets._download('https://example.org/f', destination, start=256, end=511, progress=False)

    # a download from the start of the file can still be truncated to the request
    datasets._download('https://example.org/f', destination, start=0, end=255, progress=False)
    assert open(destination, 'rb').read() == content[:256]


def _fake_urlopen_files(files, status=206):
    """``urlopen`` replacement serving several files, routed by the URL."""
    def urlopen(request, *args, **kwargs):
        url = request.full_url
        for filename, content in files.items():
            if filename in url:
                break
        else:
            raise AssertionError(f'unexpected URL {url}')
        header = request.get_header('Range')
        if header is None:
            return _FakeResponse(content, 200)
        first, last = header.replace('bytes=', '').split('-')
        stop = int(last) + 1 if last else len(content)
        return _FakeResponse(content[int(first):stop], status)
    return urlopen


FAKE_HEADER = HEADER.replace('3000', '10').replace('2999', '9')
FAKE_FRAMES = 10
FAKE_FRAME_BYTES = 4 * 5 * 2
FAKE_MRAW = (bytes(range(256)) * 2)[:FAKE_FRAMES * FAKE_FRAME_BYTES]

FAKE_DATASET = {
    'name': 'fake_recording',
    'description': 'A dataset that exists only in the tests.',
    'record': '1',
    'header_file': 'fake.cihx',
    'data_file': 'fake.mraw',
    'header_md5': hashlib.md5(FAKE_HEADER.encode()).hexdigest(),
    'n_frames_total': FAKE_FRAMES,
    'default_first_frame': 2,
    'default_n_frames': 3,
    'image_height': 4,
    'image_width': 5,
    'bytes_per_pixel': 2,
}


@pytest.fixture
def fake_dataset(monkeypatch):
    """Register a dataset for the duration of one test."""
    monkeypatch.setitem(datasets.DATASETS, FAKE_DATASET['name'], FAKE_DATASET)
    monkeypatch.setattr(
        datasets.urllib.request, 'urlopen',
        _fake_urlopen_files({'fake.cihx': FAKE_HEADER.encode(), 'fake.mraw': FAKE_MRAW}),
    )
    return FAKE_DATASET


def test_music_box_is_registered():
    assert datasets.DATASETS['music_box'] is datasets.MUSIC_BOX
    assert 'music_box' in datasets.list_datasets()


def test_register_dataset_requires_the_keys():
    with pytest.raises(ValueError, match='header_md5'):
        datasets.register_dataset({'name': 'incomplete', 'record': '1'})
    assert 'incomplete' not in datasets.DATASETS


def test_unknown_dataset_names_the_registered_ones():
    with pytest.raises(ValueError, match='music_box'):
        datasets.fetch_dataset('not_a_dataset')


def test_fetch_dataset_downloads_the_default_window(tmp_path, fake_dataset):
    cihx = datasets.fetch_dataset('fake_recording', data_dir=str(tmp_path), progress=False)

    assert os.path.basename(cihx) == 'fake_recording_f2_n3.cihx'
    mraw = cihx[: -len('.cihx')] + '.mraw'
    start = 2 * FAKE_FRAME_BYTES
    assert open(mraw, 'rb').read() == FAKE_MRAW[start: start + 3 * FAKE_FRAME_BYTES]

    patched = open(cihx, encoding='utf-8').read()
    assert '<recordedFrame>3</recordedFrame>' in patched
    assert 'Frames 2 to 4' in patched


def test_fetch_dataset_window_arguments(tmp_path, fake_dataset):
    cihx = datasets.fetch_dataset(
        'fake_recording', n_frames='all', first_frame=0, data_dir=str(tmp_path), progress=False)
    assert os.path.basename(cihx) == 'fake_recording_f0_n10.cihx'

    cihx = datasets.fetch_dataset(
        'fake_recording', n_frames=1, data_dir=str(tmp_path), progress=False)
    assert os.path.basename(cihx) == 'fake_recording_f2_n1.cihx'   # the default first_frame

    with pytest.raises(ValueError, match='fake_recording'):
        datasets.fetch_dataset(
            'fake_recording', n_frames=99, data_dir=str(tmp_path), progress=False)


def test_fetch_dataset_accepts_an_unregistered_dictionary(tmp_path, fake_dataset):
    unregistered = dict(FAKE_DATASET, name='not_registered')
    cihx = datasets.fetch_dataset(
        unregistered, n_frames=1, first_frame=0, data_dir=str(tmp_path), progress=False)
    assert os.path.basename(cihx) == 'not_registered_f0_n1.cihx'


def test_fetch_dataset_is_cached(tmp_path, fake_dataset, monkeypatch):
    datasets.fetch_dataset('fake_recording', data_dir=str(tmp_path), progress=False)

    def fail(*args, **kwargs):
        raise AssertionError('the cached window was downloaded again')

    monkeypatch.setattr(datasets.urllib.request, 'urlopen', fail)
    datasets.fetch_dataset('fake_recording', data_dir=str(tmp_path), progress=False)


@pytest.mark.skipif(
    os.environ.get('PYIDI_TEST_NETWORK') != '1',
    reason='downloads from Zenodo; set PYIDI_TEST_NETWORK=1 to run',
)
def test_load_music_box(tmp_path):
    video = datasets.load_music_box(n_frames=2, data_dir=str(tmp_path), progress=False)

    assert isinstance(video, pyidi.VideoReader)
    assert video.N == 2
    assert (video.image_height, video.image_width) == (552, 640)
    assert video.fps == 7500
    assert video.get_frame(0).shape == (552, 640)
    assert np.asarray(video.get_frame(0)).max() > 0


if __name__ == '__main__':
    pytest.main([__file__])
