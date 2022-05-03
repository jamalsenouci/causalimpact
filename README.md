## CausalImpact

[![Python package](https://github.com/jamalsenouci/causalimpact/actions/workflows/main.yml/badge.svg)](https://github.com/jamalsenouci/causalimpact/actions/workflows/main.yml)
[![Coverage Status](https://coveralls.io/repos/github/jamalsenouci/causalimpact/badge.svg)](https://coveralls.io/github/jamalsenouci/causalimpact)
![monthly downloads](https://pepy.tech/badge/causalimpact/month)

#### TO DO

- Estimation is MLE not Bayesian

#### A Python package for causal inference using Bayesian structural time-series models

This is a port of the R package CausalImpact, see: https://github.com/google/CausalImpact.

This package implements an approach to estimating the causal effect of a designed intervention on a time series. For example, how many additional daily clicks were generated by an advertising campaign? Answering a question like this can be difficult when a randomized experiment is not available. The package aims to address this difficulty using a structural Bayesian time-series model to estimate how the response metric might have evolved after the intervention if the intervention had not occurred.

As with all approaches to causal inference on non-experimental data, valid conclusions require strong assumptions. The CausalImpact package, in particular, assumes that the outcome time series can be explained in terms of a set of control time series that were themselves not affected by the intervention. Furthermore, the relation between treated series and control series is assumed to be stable during the post-intervention period. Understanding and checking these assumptions for any given application is critical for obtaining valid conclusions.

#### Try it out in the browser

https://mybinder.org/v2/gh/jamalsenouci/causalimpact/HEAD?filepath=GettingStarted.ipynb

#### Installation

install the latest release via pip

```bash
pip install causalimpact
```

#### Getting started

[Documentation and examples](https://nbviewer.org/github/jamalsenouci/causalimpact/blob/master/GettingStarted.ipynb)

#### Further resources

- Manuscript: [Brodersen et al., Annals of Applied Statistics (2015)](http://research.google.com/pubs/pub41854.html)

#### Bugs

The issue tracker is at https://github.com/jamalsenouci/causalimpact/issues. Please report any bugs that you find. Or, even better, fork the repository on GitHub and create a pull request.
