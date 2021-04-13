[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Quality][quality-image]][quality-url]

# Manifoldy

A package to generate customizable manifold synthetic datasets.

## Linting
To perform the linting of the code, run the following command:

```bash
make tox
```

Alternatively, you can start the same process without using `Make` with the following command:

```bash
python -m tox
```

Note that it is mandatory to install the dependences (tox, flake, pylint, ...):

```bash
make deps
```

[pypi-image]: https://img.shields.io/pypi/v/manifoldy
[pypi-url]: https://pypi.org/project/manifoldy/
[build-image]: https://github.com/KNODIS-Research-Group/manifoldy/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/KNODIS-Research-Group/manifoldy/actions/workflows/build.yml
[quality-image]: https://api.codeclimate.com/v1/badges/f6a3f424237d92169fc0/maintainability
[quality-url]: https://codeclimate.com/github/KNODIS-Research-Group/manifoldy
