# qfrm

[![Build Status](https://travis-ci.org/pjgranahan/qfrm_py.svg?branch=master)](https://travis-ci.org/pjgranahan/qfrm_py)
[![Code Climate](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/gpa.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py)
[![Issue Count](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/issue_count.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py)
<!--[![Test Coverage](https://codeclimate.com/github/pjgranahan/qfrm_py/badges/coverage.svg)](https://codeclimate.com/github/pjgranahan/qfrm_py/coverage)-->
<!--Not included until Code Climate supports code coverage for Python-->

Quantitative Financial Risk Management (qfrm) is a set of analytical tools to measure, manage, and visualize identified 
risks of financial derivatives and portfolios.

## What qfrm does

### Option Pricing

qfrm can price a variety of vanilla and exotic options using [Black-Scholes], [Lattice], [Finite Difference], 
and [Monte Carlo] models.

| Option Name | Black-Scholes | Lattice | Monte Carlo | Finite Difference |
|:-----------:|:-------------:|:-------:|:-----------:|:-----------------:|
| *Example* | *:white_check_mark: (supported)* | *:x: (not supported)* | *- (not applicable)* |
| **American** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| Asian | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Barrier | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| Basket | :x: | :white_check_mark: | :white_check_mark: | :x: |
| Bermudan | :x: | :white_check_mark: | :white_check_mark: | :x: |
| Binary | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| Boston | :x: | :white_check_mark: | :x: | :x: |
| Chooser | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Compound | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| ContingentPremium | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **European** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| Exchange | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ForwardStart | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Gap | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Ladder | :x: | :x: | :white_check_mark: | :white_check_mark: |
| Lookback | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| LowExercisePrice | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PerpetualAmerican | :white_check_mark: | :x: | :x: | :x: |
| Quanto | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| Rainbow | :x: | :x: | :white_check_mark: | :x: |
| Shout | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Spread | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| VarianceSwap | :white_check_mark: | :x: | :x: | :x: |

[Black-Scholes]: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
[Lattice]: https://en.wikipedia.org/wiki/Lattice_model_(finance)
[Finite Difference]: https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing
[Monte Carlo]: https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing

## Installation

To install [from PyPI] using `pip`:
```
pip install qfrm
```
[from PyPI]: https://pypi.python.org/pypi/qfrm

## Usage

TODO

## History

This project was created by undergraduate and graduate students at [Rice University] for the [Fall 2015 QFRM course] taught by Oleg Melnikov.

A [QFRM package for R] was also created during the Spring 2015 QFRM course.

The QFRM course is part of Rice University's [Center for Computational Finance and Economic Systems] (CoFES) [Financial Computation and Modeling] (FCAM) Minor for undergraduate study, led by Dr. Katherine Ensor.

Original contributors to this project:
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
[Fall 2015 QFRM course]: http://oleg.rice.edu/stat-449-649-fall-2015/
[QFRM package for R]: https://cran.r-project.org/web/packages/QFRM/
[Center for Computational Finance and Economic Systems]: http://www.cofes-rice.org/
[Financial Computation and Modeling]: http://www.cofes-rice.org/academic-programs/undergraduate/
[contributors]: https://github.com/thoughtbot/capybara-webkit/graphs/contributors

## License

TBD
