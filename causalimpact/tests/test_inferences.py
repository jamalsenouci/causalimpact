"""Unit Tests for inferences module"""


import pytest
import numpy as np
import pandas as pd
import statsmodels as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima_process import ArmaProcess


compile_posterior = causalimpact.inferences.compile_posterior_inferences


@pytest.fixture
def data():
    np.random.seed(1)
    ar = np.r_[1, 0.9]
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    
    X = 100 + arma_process.generate_sample(nsample=100)
    X = X.reshape(-1, 1)
    y = 1.2 * X + np.random.normal(size=(100, 1))
    data = np.concatenate((y, X), axis=1)
    data = pd.DataFrame(data)


def test_compile_posterior_inferences():
     
