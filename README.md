# python-multem
> Python wrapper for MULTEM

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/alerts/)

## Installation

To install from the github repository do the following

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install git+https://github.com/rosalindfranklininstitute/python-multem.git@master#egg=python-multem
```

To install from source, clone this repository and then do the following:

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
python -m pip install -r requirements.txt
python setup.py develop
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

