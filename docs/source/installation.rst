.. _installation-label:

Installation
============

.. code:: bash

    pip install pyidi

That is enough to read every supported video format and run every
identification method. The interactive point-selection and result-viewing
tools need one extra:

.. code:: bash

    pip install pyidi[qt]

Requirements
------------

pyIDI requires **Python >= 3.10**.

Everything needed for identification is installed automatically, including
``numba`` (the compiled Lucas-Kanade kernel), ``imageio[pyav]`` (video files),
``pyMRAW`` (Photron), ``cine-handler`` (Phantom ``.cine``) and
``opencv-contrib-python`` (fiducial markers).

Optional extras
---------------

``[qt]`` — the graphical tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installs napari, PyQt6, pyqtgraph and magicgui. Required for
:ref:`SelectionGUI <point-selection>`, ``ResultViewer`` and the napari
:ref:`GUI <napari>`. Without it those classes can still be imported, but
instantiating one raises a ``RuntimeError`` telling you to install the extra.

.. code:: bash

    pip install pyidi[qt]

``[dev]`` — building the docs and running the tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sphinx and its extensions, pytest, and the notebook tooling.

.. code:: bash

    pip install pyidi[dev]

Combining extras:

.. code:: bash

    pip install pyidi[qt,dev]

Upgrading
---------

.. code:: bash

    pip install -U pyidi

If you are coming from an older release, read :doc:`migration` — several
releases have changed things that will not go unnoticed.

OpenCV conflicts
----------------

pyIDI relies on ``opencv-contrib-python`` (not the base ``opencv-python``
package) for ArUco marker detection. The two packages install into the same
namespace and conflict. If you already have ``opencv-python``, remove it
first:

.. code:: bash

    pip uninstall opencv-python
    pip install pyidi

Development install
-------------------

.. code:: bash

    git clone https://github.com/ladisk/pyidi.git
    cd pyidi
    pip install -e ".[dev,qt]"

Running the tests:

.. code:: bash

    pytest

Building the documentation:

.. code:: bash

    cd docs
    make html

The result is in ``docs/build/html/index.html``.

Verifying the installation
--------------------------

.. code:: python

    import pyidi

    print(pyidi.__version__)

The first Lucas-Kanade run in a fresh environment spends a few extra seconds
compiling the numba kernel. The compiled result is cached on disk, so later
runs skip it. If pyIDI is installed somewhere the cache cannot be written, set
``NUMBA_CACHE_DIR`` to a writable directory.
