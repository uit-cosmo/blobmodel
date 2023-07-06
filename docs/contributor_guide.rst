.. _contributor-guide:

Contributor Guide
=================

Thanks for your interest in contributing to :py:mod:`blobmodel`! Any contributions such as bug reports, feature requests or even pull requests are very welcome!

+++++++++++++++++++++++++++
Install development version
+++++++++++++++++++++++++++

If you want to contribute code to :py:mod:`blobmodel`, we recommend first forking the `GitHub repository <https://github.com/uit-cosmo/blobmodel/tree/main>`_.
Next, install the package in development mode:

.. code-block:: bash

  # replace your_account with appropriate name
  git clone https://github.com/your_account/blobmodel.git 
  cd blobmodel
  pip install -e .

The :py:mod:`-e` specifies that this will be an `editable installation <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_.

.. tip::

   It is advisable to use a virtual environment for code development. Popular options are `Virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `Conda <https://docs.conda.io/en/latest/>`_.

++++++++++++++++++++++++++
Code Contribution Workflow
++++++++++++++++++++++++++

After the package is installed, we can startthe cycle of making changes:

1. Edit or create the files and save the changes.
2. In a terminal, run:

  .. code-block:: bash
   
    git add filename
    
  Replace `filename` with the name of the edited file(s).

3. Commit the changes:

  .. code-block:: bash

    git commit -m "commit message"

  Replace `commit message` with a description of the committed changes.

4. Push the changes to GitHub:

  .. code-block:: bash

    git push
          
5. If you now go back to your GitHub repository of :py:mod:`blobmodel`, a pale yellow box will appear near the top of the screen, there click on :guilabel:`Compare & pull request`.

6. Add a descriptive title and a description of the pull request, then select :guilabel:`Create pull request`.  

++++++++++++++++
Formatting Guide
++++++++++++++++

:py:mod:`blobmodel` uses `Black <https://github.com/psf/black>`_ for code formatting. Make sure to run 

.. code-block:: bash

   black edited/new file

on all edited and added python files in your pull request. For documentation we recomend following the `numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

+++++++++++++
Testing Guide
+++++++++++++

Any new code contributed to :py:mod:`blobmodel` requires testing before it can be merged. All tests are located in the subdirectory `tests/`. 
After you added your tests to this directory you can run

.. code-block:: bash
  
   pytest

to check whether all tests pass. 

In order to check whether all your new code is covered by tests, run:

.. code-block:: bash

   pytest --cov
   coverage html

You can now open `htmlcov/index.html` with your browser to check whether all of your lines are covered by tests.

As a last point, we recommend adding type hints to your new functions and classes, which ensures that you’re using variables and functions in your code correctly. 
We use `mypy <https://mypy.readthedocs.io/en/stable/>`_ for this purpose. Check your code by running:

.. code-block:: bash

   mypy  --ignore-missing-imports .
