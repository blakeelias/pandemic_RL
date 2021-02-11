import pickle
from pathlib import Path
from scipy.stats import rv_discrete
import numpy as np
from pdb import set_trace as b

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
    probs = distribution.pmf(support)
    low = 0
    high = len(feasible_range) - 1
    probs[low] = distribution.pmf(support[low]) + distribution.cdf(support[low] - 1)
    probs[high] = 1 - distribution.cdf(support[high] - 1)

    try:
        return rv_discrete(values=(support, probs))
    except:
        b()


class CappedDistribution:
'''
>>> from scipy.stats import poisson
>>> poisson(5)
<scipy.stats._distn_infrastructure.rv_frozen object at 0x7f97d80df340>
>>> poisson(5).pmf(range(10))
array([0.00673795, 0.03368973, 0.08422434, 0.1403739 , 0.17546737,
       0.17546737, 0.14622281, 0.10444486, 0.06527804, 0.03626558])
>>> from utils import CappedDistribution
>>> d = CappedDistribution(poisson(5), range(1, 9))
>>> d.pmf(range(10))
array([0.        , 0.04042768, 0.08422434, 0.1403739 , 0.17546737,
       0.17546737, 0.14622281, 0.10444486, 0.13337167, 0.        ])
>>> d.pmf(range(15))
array([0.        , 0.04042768, 0.08422434, 0.1403739 , 0.17546737,
       0.17546737, 0.14622281, 0.10444486, 0.13337167, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])
>>> poisson(5).pmf(range(15))
array([0.00673795, 0.03368973, 0.08422434, 0.1403739 , 0.17546737,
       0.17546737, 0.14622281, 0.10444486, 0.06527804, 0.03626558,
       0.01813279, 0.00824218, 0.00343424, 0.00132086, 0.00047174])
'''

    def __init__(self, distribution, feasible_range):
        self.distribution = distribution
        self.feasible_range = feasible_range
        self.capped_distribution = None
        
    def rvs(self):
        # TODO: accept (*args, **kwargs) [e.g. to support `size` parameter]
        candidate = self.distribution.rvs()
        ceiling = min(candidate, max(self.feasible_range))
        floor = max(ceiling, min(self.feasible_range))
        return floor

    def pmf(self, k):
        if not self.capped_distribution:
            self.capped_distribution = cap_distribution(self.distribution, self.feasible_range)
        return self.capped_distribution.pmf(k)
