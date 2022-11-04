# python-multem
> Python wrapper for MULTEM

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/alerts/)
[![Building](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-package.yml/badge.svg)](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-package.yml)
[![Publishing](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-publish.yml)
[![DOI](https://zenodo.org/badge/201027691.svg)](https://zenodo.org/badge/latestdoi/201027691)

## Installation

To install from the github repository do the following

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install git+https://github.com/rosalindfranklininstitute/python-multem.git@master
```

To install from source, clone this repository. The repository has a submodule
for pybind11 so after cloning the repository run

```sh
git submodule update --init --recursive
```

Then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .
```

If you would like to run the tests then, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .[test]
```

## Installation for developers

To install for development, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install -e .
```

## Testing

To run the tests, follow the installation instructions and execute the following:

```sh
pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/python-multem/issues) to submit bugs or request features.

## License

Copyright Diamond Light Source, 2019.

Distributed under the terms of the GPLv3 license, python-multem is free and open source software.

