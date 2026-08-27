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
