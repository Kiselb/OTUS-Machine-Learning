import numpy as np
import pandas as pd
import os
from scipy import signal

from sklearn.datasets import fetch_california_housing
#from sklearn.datasets import load_boston
#ImportError: 
#`load_boston` has been removed from scikit-learn since version 1.2.
#
#The Boston housing prices dataset has an ethical problem: as
#investigated in [1], the authors of this dataset engineered a
#non-invertible variable "B" assuming that racial self-segregation had a
#positive impact on house prices [2]. Furthermore the goal of the
#research that led to the creation of this dataset was to study the
#impact of air quality but it did not give adequate demonstration of the
#validity of this assumption.
#
#The scikit-learn maintainers therefore strongly discourage the use of
#this dataset unless the purpose of the code is to study and educate
#about ethical issues in data science and machine learning.
#
#In this special case, you can fetch the dataset from the original
#source::
#
#    import pandas as pd
#    import numpy as np
#
#    data_url = "http://lib.stat.cmu.edu/datasets/boston"
#    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#    target = raw_df.values[1::2, 2]
#
#Alternative datasets include the California housing dataset and the
#Ames housing dataset. You can load the datasets as follows::
#
#    from sklearn.datasets import fetch_california_housing
#    housing = fetch_california_housing()
#
#for the California housing dataset and::
#
#    from sklearn.datasets import fetch_openml
#    housing = fetch_openml(name="house_prices", as_frame=True)
#
#for the Ames housing dataset.
#
#[1] M Carlisle.
#"Racist data destruction?"
#<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>
#
#[2] Harrison Jr, David, and Daniel L. Rubinfeld.
#"Hedonic housing prices and the demand for clean air."
#Journal of environmental economics and management 5.1 (1978): 81-102.
#<https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.datasets import make_blobs

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=bool) #dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (np.sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y


def load_extended_boston():
    boston = fetch_california_housing() #load_boston()
    X = boston.data

    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target


def load_citibike():
    data_mine = pd.read_csv(os.path.join(DATA_PATH, "citibike.csv"))
    data_mine['one'] = 1
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    data_starttime = data_mine.set_index("starttime")
    data_resampled = data_starttime.resample("3h").sum().fillna(0)
    return data_resampled.one


def make_signals():
    # fix a random state seed
    rng = np.random.RandomState(42)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    # create three signals
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    # concatenate the signals, add noise
    S = np.c_[s1, s2, s3]
    S += 0.2 * rng.normal(size=S.shape)

    S /= S.std(axis=0)  # Standardize data
    S -= S.min()
    return S
