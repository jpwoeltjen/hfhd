import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hfhd import sim
from hfhd import hf
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    # generate a realization of prices
    factor_loadings = np.column_stack(
            (np.arange(10, 15)/10, np.flip(np.arange(10, 15)/10)))
    ind_loadings = np.zeros(factor_loadings.shape)
    u = sim.Universe(0.01,
                     [0.000000001, 0, 0.48, 0.5, 0.00000001],
                     [0.000000001, 0, 0.48, 0.5, 0.00000001],
                     [0.000000001, 0, 0.48, 0.5, 0.00000001],
                     factor_loadings,
                     ind_loadings,
                     0.5,
                     100,
                     'm')
    u.simulate(1)
    u.price.to_pickle(f'mock_prices_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pkl')
    pt = hf.refresh_time([u.price[c].dropna() for c in u.price.columns])
    pt.to_pickle(f'mock_previous_ticks_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pkl')
    print(u.price.head(15))
    print(pt.head(15))
    print(u.price.tail(15))
    print(pt.tail(15))
