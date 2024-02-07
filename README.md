[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]

# janus-core

Tools for machine learnt interatomic potentials

## Development

1. Install [poetry](https://python-poetry.org/docs/#installation)
2. (Optional) Create a virtual environment
3. Install `janus-core` with dependencies:

```shell
git clone https://github.com/stfc/janus-core
cd janus-core
pip install --upgrade pip
poetry install --with pre-commit,dev,docs  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

## License

[BSD 3-Clause License](LICENSE)

## Funding

Contributors to this project were funded by

[![PSDI](docs/source/images/psdi-100.webp)](https://www.psdi.ac.uk/)
[![ALC](docs/source/images/alc-100.webp)](https://adalovelacecentre.ac.uk/)
[![CoSeC](docs/source/images/cosec-100.webp)](https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx)


[ci-badge]: https://github.com/stfc/janus-core/workflows/ci/badge.svg
[ci-link]: https://github.com/stfc/janus-core/actions
[cov-badge]: https://coveralls.io/repos/github/stfc/janus-core/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/stfc/janus-core?branch=main
[docs-badge]: https://github.com/stfc/janus-core/actions/workflows/docs.yml/badge.svg
[docs-link]: https://stfc.github.io/janus-core/
[pypi-badge]: https://badge.fury.io/py/janus-core.svg
[pypi-link]: https://badge.fury.io/py/janus-core
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
