===========================================================
scTM: A pacakge for topic modelling in transcriptomics data
===========================================================

.. image:: https://img.shields.io/pypi/v/sctm.svg
        :target: https://pypi.org/project/scTM


.. image:: https://readthedocs.org/projects/sctm/badge/?version=latest
        :target: https://JinmiaoChenLab.github.io/scTM/
        :alt: Documentation Status



scTM is a package for spatial transcriptomics for single cell that uses topic modelling, solved with stochastic variational infernce. The interesting
part is with the formulation of topic models, we can get interpretable embedding which are useful for downstream analysis.

Currently available modules: STAMP

* Free software: MIT license
* Documentation: https://JinmiaoChenLab.github.io/scTM/.


Features
--------

- STAMP: A spatially-aware dimensional reduction designed for spatial data.

Minimal Installation
--------------------

.. code-block:: python

    conda create --name sctm python=3.8
    conda activate sctm
    pip install sctm

or

.. code-block:: python

    conda create --name sctm python=3.8
    git clone https://github.com/JinmiaoChenLab/scTM.git
    conda activate sctm
    cd scTM
    pip install .

Basic Usage
-----------
Check out our usage of STAMP in the documentation with a simulated data at https://jinmiaochenlab.github.io/scTM/notebooks/stamp/simulation.The simulated
data can be found under data/simulation.h5ad.
For more advanced usage please check the examples in our detailed tutorials.

Tested with
-----------
python == 3.8
numpy == 1.24.4
numba == 0.57.1
torch == 2.0.3

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
