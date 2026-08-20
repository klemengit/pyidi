.. _documenting-label:

Contributing
============

Development install
-------------------

.. code:: bash

    git clone https://github.com/ladisk/pyidi.git
    cd pyidi
    pip install -e ".[dev,qt]"

    pytest                                              # tests
    flake8 . --max-line-length=127 --max-complexity=10  # lint, as in CI

Adding a displacement identification method
-------------------------------------------

1. Create ``pyidi/methods/_name_of_method.py``.
2. Inherit from :class:`~pyidi.methods.idi_method.IDIMethod`.
3. Implement ``configure()`` and ``calculate_displacements()``.
4. Export the class in ``pyidi/methods/__init__.py``.
5. Add an ``automodule`` entry in ``docs/source/code/modules.rst`` and a
   section in ``docs/source/quick_start/disp_id_methods.rst``.

.. important::

    **Every parameter of ``configure()`` must be stored as an attribute of the
    same name.**

    .. code:: python

        def configure(self, param1=None, param2=None):
            if param1 is not None:
                self.param1 = param1
            if param2 is not None:
                self.param2 = param2

    This is not a style preference. The settings dictionary, the JSON export,
    the checkpoint comparison that decides whether an interrupted analysis can
    be resumed, and ``load_analysis()`` all work by reading the attributes
    named after the ``configure()`` signature. A parameter stored under a
    different name silently drops out of all four.

``calculate_displacements()`` must set ``self.displacements`` with shape
``(n_points, n_frames, 2)``, in ``(row, column)`` order.

Building the documentation
--------------------------

.. code:: bash

    cd docs
    make html

The result is in ``docs/build/html/index.html``. The build should be free of
warnings; a broken cross-reference is a warning, so this is worth checking
before opening a pull request.

The documentation is built by Sphinx with the ``sphinx-book-theme``,
``sphinx-copybutton``, ``sphinx-design`` (the cards on the landing page) and
``myst-parser`` (which lets ``CHANGELOG.md`` be included directly). Read the
Docs builds it from ``docs/requirements.txt``, so a new extension has to be
added there as well as to the ``dev`` extra in ``pyproject.toml``.

Automatic code documentation with autodoc
-----------------------------------------

The Sphinx autodoc_ extension pulls the docstrings out of the source. Only
modules listed in ``docs/source/code/modules.rst`` are included — there is no
point documenting every internal module, so the list covers the public API and
the classes developers extend.

.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

To add a module:

.. code:: rst

    Tools
    -----

    .. automodule:: pyidi.tools
        :members:

Docstring style
---------------

pyIDI uses **reStructuredText (Sphinx) style** docstrings. Some modules use
NumPy or Google style; ``napoleon`` is enabled so those render correctly too,
but new code should match the surrounding style, which for most of the package
is reStructuredText.

* It is the default docstring style in PyCharm_.
* In VSCode, set ``"autoDocstring.docstringFormat": "sphinx"`` in the
  `VSCode Python Docstring extension`_.

.. _PyCharm: https://www.jetbrains.com/help/pycharm/python-integrated-tools.html
.. _`VSCode Python Docstring extension`: https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring

The order of the fields is: parameters (``:param <name>:``), their types
(``:type <name>:``), the return value (``:return:``, ``:rtype:``), then any
``.. note::``, ``.. warning::`` or ``.. seealso::`` directives.

.. code-block:: python

    def function1(self, arg1, arg2, arg3):
        """Return ``(arg1 / arg2) + arg3``.

        A longer explanation, which may include maths in latex syntax
        :math:`\\alpha`.

        :param arg1: the first value
        :type arg1: int or float
        :param arg2: the second value, must be non-zero
        :type arg2: int or float
        :param arg3: the third value
        :type arg3: int or float
        :return: ``arg1 / arg2 + arg3``
        :rtype: float

        .. warning:: ``arg2`` must be non-zero.
        """
        return arg1 / arg2 + arg3

Two things trip up the build regularly:

* a continuation line of a field must be indented further than the ``:param:``
  it belongs to, otherwise docutils reports *"Field list ends without a blank
  line"*;
* a directive (``.. list-table::``, ``.. code::``) swallows every following
  line that is indented at least as far as its content, so a block quote after
  a table needs an unindented line between them.

Releasing
---------

.. code:: bash

    python sync_version.py --bump patch     # or minor / major
    git tag vX.Y.Z && git push origin master --tags

``sync_version.py`` keeps the version consistent across ``pyproject.toml``,
``pyidi/__init__.py`` and ``docs/source/conf.py``. CI publishes to PyPI on a
tag push. Record user-visible changes in ``CHANGELOG.md`` — it is rendered
into the documentation as the :doc:`../changelog` page.
