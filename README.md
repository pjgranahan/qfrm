# qfrm

[![Build Status](https://travis-ci.org/pjgranahan/qfrm_py.svg?branch=master)](https://travis-ci.org/pjgranahan/qfrm_py)
[![Code Climate](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/gpa.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py)
[![Test Coverage](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/coverage.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py/coverage)
[![Issue Count](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/issue_count.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py)

Quantitative Financial Risk Management (qfrm) qfrm is a set of analytical tools to measure, manage, and visualize identified risks of financial derivatives and portfolios.

## What qfrm does

### Option Pricing

qfrm can price a variety of vanilla and exotic options using [Black-Scholes], [Lattice], [Finite Difference], and [Monte Carlo] models.

| Option Name | Black-Scholes | Lattice | Finite Difference | Monte Carlo |
|:-----------:|:-------------:|:-------:|:-----------------:|:-----------:|
| *Example* | *:white_check_mark: (supported)* | *:x: (not supported)* | *- (not applicable)* |
| American | :white_check_mark: | :white_check_mark: | :white_check_mark: | - |
| European     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Binary | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

[Black-Scholes]: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
[Lattice]: https://en.wikipedia.org/wiki/Lattice_model_(finance)
[Finite Difference]: https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing
[Monte Carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing

## Installation

The qfrm package is hosted on PyPi: https://pypi.python.org/pypi/qfrm

To install using `pip`:
```
pip install qfrm
```

## Usage

TODO

## Contributors

qfrm was originally created by undergraduate and graduate students at [Rice University] for the [QFRM course] taught by Oleg Melnikov.

- [Oleg Melnikov](https://github.com/omelnikov)
- Thaw Da Aung
- Yen-Fei Chen
- [Patrick Granahan](https://github.com/pjgranahan)
- Hanting Li
- Sha (Andy) Liao
- Scott Morgan
- Andrew M. Weatherly
- Mengyan Xie
- Tianyi Yao
- Runmin Zhang

See [contributors] for a full list of contributors. Thank you to all contributors!

[Rice University]: http://www.rice.edu/
[QFRM course]: http://oleg.rice.edu/stat-449-649-fall-2015/
[contributors]: https://github.com/thoughtbot/capybara-webkit/graphs/contributors

## License

TBD
