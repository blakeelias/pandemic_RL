import pickle
from pathlib import Path
from scipy.stats import rv_discrete
import numpy as np

def save_pickle(obj, file_name):
    Path(Path(file_name).parent).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'w') as f:
        s = pickle.dumps(obj, protocol=0)
        f.write(s.decode('latin-1'))

def load_pickle(file_name):
    with open(file_name, 'r') as f:
        s = f.read()
        obj = pickle.loads(s.encode('latin-1'))
    return obj


def combine_dicts(a: dict, b: dict):
    c = dict(a)
    c.update(b)
    return c


def cap_distribution(distribution, feasible_range):
    '''
    >>> from scipy.stats import poisson
    >>> from utils import cap_distribution
    >>> distribution = poisson(3)
    >>> feasible_range = range(6)
    >>> capped = cap_distribution(distribution, feasible_range)
    >>> distribution.pmf(range(10))
    array([0.04978707, 0.14936121, 0.22404181, 0.22404181, 0.16803136,
           0.10081881, 0.05040941, 0.02160403, 0.00810151, 0.0027005 ])
    >>> capped.pmf(range(10))
    array([0.04978707, 0.14936121, 0.22404181, 0.22404181, 0.16803136,
           0.18473676, 0.        , 0.        , 0.        , 0.        ])
    >>> feasible_range = range(1, 6)
    >>> capped = cap_distribution(distribution, feasible_range)
    >>> capped.pmf(range(10))
    array([0.        , 0.19914827, 0.22404181, 0.22404181, 0.16803136,
           0.18473676, 0.        , 0.        , 0.        , 0.        ])
    '''
    # feasible_range = list(feasible_range)
    support = feasible_range
    probs = np.zeros(len(feasible_range))
    for i, x in enumerate(feasible_range):
        if i == 0:
            probs[i] = distribution.pmf(x) + distribution.cdf(x - 1)
        elif i == len(feasible_range) - 1:
            probs[i] = 1 - distribution.cdf(x - 1)
        else:
            probs[i] = distribution.pmf(x)

    return rv_discrete(values=(support, probs))
