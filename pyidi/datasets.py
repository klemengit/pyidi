"""
Example datasets, downloaded from Zenodo on first use.

The datasets are high-speed camera recordings that are too large to ship with
the package, so they are fetched on demand and cached on disk. Only the frames
that are asked for are downloaded (Zenodo serves HTTP range requests), which
keeps the first call to a few hundred MB instead of the full recording.

Example::

    import pyidi

    pyidi.datasets.list_datasets()                # what is available
    video = pyidi.datasets.load_dataset('music_box')   # downloads on first call
    video = pyidi.datasets.load_music_box()            # the same, by name
    lk = pyidi.LucasKanade(video)

The cache directory is ``~/.pyidi/datasets`` and can be redirected with the
``PYIDI_DATA_DIR`` environment variable or the ``data_dir`` argument.

Adding a dataset
----------------

A dataset is a dictionary of metadata in the :data:`DATASETS` registry; no code
is needed to add one. :func:`register_dataset` accepts one from anywhere, so a
recording that is not part of pyidi can be loaded through the same functions::

    pyidi.datasets.register_dataset({
        'name': 'my_recording',
        'record': '1234567',            # Zenodo record id
        'header_file': 'my_recording.cihx',
        'data_file': 'my_recording.mraw',
        'header_md5': '...',
        'n_frames_total': 5000,
        'image_height': 512,
        'image_width': 512,
        'bytes_per_pixel': 2,
    })
    video = pyidi.datasets.load_dataset('my_recording')

The download strategy assumes what every dataset published this way has in
common: a Zenodo record holding a Photron ``cihx`` header next to an
uncompressed ``mraw`` file of fixed-size frames. Frames are addressed by byte
offset, which is what makes fetching a window of a 36 GiB recording possible,
and the header is rewritten to describe the downloaded window.

@author: Janko Slavič (janko.slavic@fs.uni-lj.si)
"""

import hashlib
import os
import re
import urllib.request

from tqdm import tqdm

from .video_reader import VideoReader

__all__ = [
    "load_dataset", "fetch_dataset", "list_datasets", "register_dataset",
    "load_music_box", "fetch_music_box", "get_data_dir",
    "DATASETS", "MUSIC_BOX", "REQUIRED_KEYS",
]

#: Keys every dataset in the registry must define. The remaining keys
#: (``doi``, ``url``, ``data_md5``, ``fps``, ``license``, ``citation``,
#: ``description``, ``default_first_frame``, ``default_n_frames``) are optional.
REQUIRED_KEYS = (
    "name", "record", "header_file", "data_file", "header_md5",
    "n_frames_total", "image_height", "image_width", "bytes_per_pixel",
)

#: The registered datasets, by name. See :func:`register_dataset`.
DATASETS = {}

#: Metadata of the music-box recording published at
#: https://doi.org/10.5281/zenodo.22105821.
MUSIC_BOX = {
    "name": "music_box",
    "description": (
        "Vibrating comb of a mechanical music box, Photron FASTCAM SA-Z, "
        "7500 fps, 640x552 px, 12-bit."
    ),
    "doi": "10.5281/zenodo.22105821",
    "record": "22105821",
    "url": "https://doi.org/10.5281/zenodo.22105821",
    "header_file": "music_box_excerpt.cihx",
    "data_file": "music_box_excerpt.mraw",
    "header_md5": "d31811b6c95d5c0191de73668149c4a1",
    "data_md5": "08e3c3007e6b880bf63254d6ccc4b81a",
    "n_frames_total": 3000,    # frames in the published excerpt
    "default_first_frame": 400,  # after the pluck, where both teeth ring freely
    "default_n_frames": 600,
    "image_height": 552,
    "image_width": 640,
    "bytes_per_pixel": 2,      # 16-bit container, 12 effective bits
    "fps": 7500,
    "license": "CC BY 4.0",
    "citation": (
        "Stanovnik, G., & Slavič, J. (2026). High-speed video of a vibrating "
        "music-box comb (Photron FASTCAM SA-Z, 7500 fps, 640x552 px) [Data set]. "
        "Zenodo. https://doi.org/10.5281/zenodo.22105821"
    ),
}

ZENODO_FILE_URL = "https://zenodo.org/records/{record}/files/{filename}?download=1"

_CHUNK = 1 << 20


def register_dataset(dataset):
    """Add a dataset to the :data:`DATASETS` registry.

    The dataset is a dictionary of metadata; see :data:`REQUIRED_KEYS` for the
    keys it must define and :data:`MUSIC_BOX` for a complete example. A name
    that is already registered is replaced.

    :param dataset: metadata of the dataset
    :type dataset: dict
    :return: the registered dataset
    :rtype: dict
    """
    missing = [key for key in REQUIRED_KEYS if key not in dataset]
    if missing:
        raise ValueError(f"The dataset is missing the keys: {', '.join(missing)}.")
    DATASETS[dataset["name"]] = dataset
    return dataset


def list_datasets():
    """Names and descriptions of the registered datasets.

    :return: description of each dataset, by name
    :rtype: dict
    """
    return {name: dataset.get("description", "") for name, dataset in sorted(DATASETS.items())}


def _get_dataset(name):
    """Look a dataset up in the registry, by name or as a metadata dictionary."""
    if isinstance(name, dict):
        return name
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. The registered datasets are: "
            f"{', '.join(sorted(DATASETS))}."
        ) from None


def get_data_dir(data_dir=None):
    """Directory where the downloaded datasets are cached.

    The directory is created if it does not exist.

    :param data_dir: cache directory. If None, the ``PYIDI_DATA_DIR``
        environment variable is used, and if that is not set either,
        ``~/.pyidi/datasets``. Defaults to None.
    :type data_dir: str, optional
    :return: path of the cache directory
    :rtype: str
    """
    if data_dir is None:
        data_dir = os.environ.get(
            "PYIDI_DATA_DIR", os.path.join(os.path.expanduser("~"), ".pyidi", "datasets")
        )
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _md5(path):
    """MD5 checksum of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, destination, start=0, end=None, progress=True, label=None):
    """Download a file, or a byte range of it, resuming a partial download.

    The bytes are written to ``destination + '.part'`` and moved into place only
    when the expected number of bytes has arrived, so an interrupted download
    never leaves a truncated file behind.

    :param url: source URL
    :type url: str
    :param destination: path of the downloaded file
    :type destination: str
    :param start: first byte to download, defaults to 0
    :type start: int, optional
    :param end: last byte to download (inclusive). If None, the file is
        downloaded to its end. Defaults to None.
    :type end: int, optional
    :param progress: show a progress bar, defaults to True
    :type progress: bool, optional
    :param label: progress bar description, defaults to None
    :type label: str, optional
    """
    partial = destination + ".part"
    expected = None if end is None else end - start + 1

    have = os.path.getsize(partial) if os.path.exists(partial) else 0
    if expected is not None and have >= expected:
        # left over from a different (larger) request
        have = 0
    if have:
        print(f"Resuming the download of {os.path.basename(destination)} at {have} bytes.")

    request = urllib.request.Request(url)
    ranged = start + have > 0 or end is not None
    if ranged:
        stop = "" if end is None else str(end)
        request.add_header("Range", f"bytes={start + have}-{stop}")

    with urllib.request.urlopen(request) as response:
        if ranged and response.status != 206:
            # the server ignored the range; only a download from scratch can be salvaged
            if start + have > 0:
                raise ConnectionError(
                    f"{url} does not support range requests, cannot download frames "
                    f"starting at byte {start}."
                )
            have = 0

        total = expected
        if total is None:
            length = response.headers.get("Content-Length")
            total = int(length) + have if length is not None else None

        with open(partial, "ab" if have else "wb") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
            desc=label or os.path.basename(destination), disable=not progress,
        ) as bar:
            written = have
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                if expected is not None and written + len(chunk) > expected:
                    chunk = chunk[: expected - written]  # server ignored the range end
                f.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
                if expected is not None and written >= expected:
                    break

    if expected is not None and os.path.getsize(partial) != expected:
        raise ConnectionError(
            f"Downloaded {os.path.getsize(partial)} bytes from {url}, expected {expected}."
        )
    os.replace(partial, destination)


def _fetch_header(dataset, data_dir, progress=True):
    """Download the original ``cihx`` header of a dataset (a few tens of kB)."""
    path = os.path.join(data_dir, dataset["header_file"])
    if not os.path.exists(path) or _md5(path) != dataset["header_md5"]:
        _download(
            ZENODO_FILE_URL.format(record=dataset["record"], filename=dataset["header_file"]),
            path, progress=progress, label=dataset["header_file"],
        )
        if _md5(path) != dataset["header_md5"]:
            raise ConnectionError(f"Checksum mismatch of the downloaded {dataset['header_file']}.")
    return path


def _patch_header(header, dataset, first_frame, n_frames):
    """Rewrite the frame counts of a ``cihx`` header for a subset of frames.

    The downloaded ``mraw`` file holds ``n_frames`` frames, so the header must
    say so; otherwise the reader looks for frames that are not in the file.

    :param header: contents of the original cihx file
    :type header: str
    :param dataset: dataset metadata dictionary
    :type dataset: dict
    :param first_frame: index of the first frame in the subset
    :type first_frame: int
    :param n_frames: number of frames in the subset
    :type n_frames: int
    :return: contents of the cihx file describing the subset
    :rtype: str
    """
    frame_info = re.search(r"<frameInfo>.*?</frameInfo>", header, flags=re.S)
    if frame_info is None:
        raise ValueError("The cihx header has no <frameInfo> section.")

    block = frame_info.group()
    for tag, value in (
        ("recordedFrame", n_frames),
        ("totalFrame", n_frames),
        ("startFrame", 0),
        ("playbackStartFrameNo", 0),
        ("playbackEndFrameNo", n_frames - 1),
        ("saveStartFrameNo", 0),
        ("saveEndFrameNo", n_frames - 1),
    ):
        block, count = re.subn(rf"<{tag}>[^<]*</{tag}>", f"<{tag}>{value}</{tag}>", block)
        if count != 1:
            raise ValueError(f"Expected one <{tag}> tag in <frameInfo>, found {count}.")
    header = header[: frame_info.start()] + block + header[frame_info.end():]

    last = first_frame + n_frames - 1
    source = dataset.get("doi") or dataset.get("url") or f"Zenodo record {dataset['record']}"
    comment = (
        f"Frames {first_frame} to {last} of {dataset['data_file']} "
        f"({source}), downloaded by pyidi.datasets."
    )
    header, count = re.subn(r"<comment>[^<]*</comment>", f"<comment>{comment}</comment>", header)
    if count == 0:
        header = header.replace("</basicInfo>", f"    <comment>{comment}</comment>\n        </basicInfo>")
    return header


def _resolve_window(dataset, n_frames, first_frame):
    """Resolve the requested window against the defaults of a dataset.

    :param dataset: dataset metadata dictionary
    :type dataset: dict
    :param n_frames: number of frames, None for the default of the dataset and
        ``'all'`` for the whole file
    :type n_frames: int, 'all' or None
    :param first_frame: index of the first frame, None for the default of the
        dataset
    :type first_frame: int or None
    :return: first frame and number of frames
    :rtype: tuple of int
    """
    available = dataset["n_frames_total"]
    if first_frame is None:
        first_frame = dataset.get("default_first_frame", 0)
    if n_frames is None:
        n_frames = dataset.get("default_n_frames", available - first_frame)
    elif n_frames == "all":
        n_frames = available - first_frame

    if first_frame < 0 or n_frames < 1 or first_frame + n_frames > available:
        raise ValueError(
            f"The '{dataset['name']}' dataset has {available} frames, cannot read "
            f"{n_frames} frames from frame {first_frame}."
        )
    return first_frame, n_frames


def fetch_dataset(name, n_frames=None, first_frame=None, data_dir=None, progress=True, force=False):
    """Download a window of a dataset and return the path to its header file.

    Only the requested frames are downloaded and they are cached, so the second
    call is free. Each window is cached under its own name, so asking for a
    different window downloads it next to the first one rather than replacing
    it.

    :param name: name of a registered dataset, see :func:`list_datasets`. A
        metadata dictionary is also accepted, for a dataset that is not
        registered.
    :type name: str or dict
    :param n_frames: number of frames to download. If None, the default window
        of the dataset is used, and ``'all'`` downloads the file to its end.
        Defaults to None.
    :type n_frames: int, 'all' or None, optional
    :param first_frame: index of the first downloaded frame. If None, the
        default of the dataset is used. Defaults to None.
    :type first_frame: int, optional
    :param data_dir: cache directory, see :func:`get_data_dir`. Defaults to None.
    :type data_dir: str, optional
    :param progress: show a progress bar, defaults to True
    :type progress: bool, optional
    :param force: download again even if the files are already cached,
        defaults to False
    :type force: bool, optional
    :return: path of the cached ``cihx`` file (the ``mraw`` file is next to it)
    :rtype: str
    """
    dataset = _get_dataset(name)
    first_frame, n_frames = _resolve_window(dataset, n_frames, first_frame)

    data_dir = get_data_dir(data_dir)
    frame_bytes = dataset["image_height"] * dataset["image_width"] * dataset["bytes_per_pixel"]
    cache_name = f"{dataset['name']}_f{first_frame}_n{n_frames}"
    cihx_path = os.path.join(data_dir, cache_name + ".cihx")
    mraw_path = os.path.join(data_dir, cache_name + ".mraw")
    complete = (
        os.path.exists(cihx_path)
        and os.path.exists(mraw_path)
        and os.path.getsize(mraw_path) == n_frames * frame_bytes
    )

    if complete and not force:
        return cihx_path

    if progress:
        size = n_frames * frame_bytes / 1024**2
        source = dataset.get("doi") or dataset.get("url") or f"Zenodo record {dataset['record']}"
        licence = dataset.get("license")
        print(
            f"Downloading {n_frames} frames ({size:.0f} MiB) of the {dataset['name']} recording "
            f"({source}{', ' + licence if licence else ''}) to {data_dir}."
        )

    original_header = _fetch_header(dataset, data_dir, progress=progress)
    with open(original_header, "r", encoding="utf-8", errors="ignore") as f:
        header = f.read()

    if force:
        for path in (mraw_path, mraw_path + '.part'):
            if os.path.exists(path):
                os.remove(path)
    _download(
        ZENODO_FILE_URL.format(record=dataset["record"], filename=dataset["data_file"]),
        mraw_path,
        start=first_frame * frame_bytes,
        end=(first_frame + n_frames) * frame_bytes - 1,
        progress=progress,
        label=f"{n_frames} frames",
    )

    whole_file = first_frame == 0 and n_frames == dataset["n_frames_total"]
    if whole_file and dataset.get("data_md5") and _md5(mraw_path) != dataset["data_md5"]:
        raise ConnectionError(f"Checksum mismatch of the downloaded {dataset['data_file']}.")

    with open(cihx_path, "w", encoding="utf-8") as f:
        f.write(_patch_header(header, dataset, first_frame, n_frames))
    return cihx_path


def load_dataset(name, n_frames=None, first_frame=None, data_dir=None, progress=True, force=False):
    """Load a dataset, downloading it on first use.

    See :func:`list_datasets` for what is available and :func:`fetch_dataset`
    for the arguments, which are passed on.

    :return: reader of the downloaded recording
    :rtype: :class:`pyidi.video_reader.VideoReader`
    """
    cihx_path = fetch_dataset(
        name, n_frames=n_frames, first_frame=first_frame, data_dir=data_dir,
        progress=progress, force=force,
    )
    return VideoReader(cihx_path)


def fetch_music_box(n_frames=None, first_frame=None, data_dir=None, progress=True, force=False):
    """Download the music-box recording and return the path to its header file.

    The full recording (36 GiB) and the excerpt used here (3000 frames, 2.0 GiB)
    are published at https://doi.org/10.5281/zenodo.22105821. Only the requested
    frames are downloaded and they are cached, so the second call is free. One
    frame is 640x552 px of 16-bit data, i.e. 0.67 MiB; the 600 frames of the
    default window are 404 MiB.

    In the excerpt, one tooth is ringing from the start and another one is
    plucked at about frame 350. The default window (600 frames from frame 400)
    opens after that pluck, where both teeth ring freely and the motion is
    smooth enough to be tracked from a single reference image. Use
    ``first_frame=0`` to include the pluck itself, which is a much harder case:
    the tooth then moves by more than 10 px between consecutive frames.

    This is :func:`fetch_dataset` with ``name='music_box'``; see it for the
    arguments, which are passed on. ``n_frames='all'`` downloads the excerpt to
    its end.

    :return: path of the cached ``cihx`` file (the ``mraw`` file is next to it)
    :rtype: str
    """
    return fetch_dataset(
        MUSIC_BOX["name"], n_frames=n_frames, first_frame=first_frame,
        data_dir=data_dir, progress=progress, force=force,
    )


def load_music_box(n_frames=None, first_frame=None, data_dir=None, progress=True, force=False):
    """Load the music-box recording, downloading it on first use.

    A high-speed video of the vibrating steel comb of a mechanical music box,
    recorded with a Photron FASTCAM SA-Z at 7500 fps, 640x552 px, 12-bit. The
    teeth are cantilevers of graduated length and each rings at its own natural
    frequencies, with sub-pixel amplitudes on a naturally speckled surface,
    which makes the recording a convenient benchmark for displacement
    identification. Published under CC BY 4.0 at
    https://doi.org/10.5281/zenodo.22105821.

    Example::

        video = pyidi.datasets.load_music_box()

        lk = pyidi.LucasKanade(video)
        lk.set_points([[109, 500], [175, 500], [329, 500]])   # three teeth
        lk.configure(roi_size=(21, 51))       # one tooth tall, wide enough
        displacements = lk.get_displacements()

    See :func:`fetch_music_box` for the arguments, which are passed on.

    :return: reader of the downloaded recording
    :rtype: :class:`pyidi.video_reader.VideoReader`
    """
    cihx_path = fetch_music_box(
        n_frames=n_frames, first_frame=first_frame, data_dir=data_dir,
        progress=progress, force=force,
    )
    return VideoReader(cihx_path)


register_dataset(MUSIC_BOX)
