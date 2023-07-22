# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import numpy as np
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization
from ..utils.utils_ import load_pkl

class Fathalla_initialization(Initialization):
    def __init__(self,
                 sampling=None,
                 repair=None,
                 eliminate_duplicates=None,
                 pkl_data= None) -> None:
        self.pkl_data = pkl_data
        super().__init__(sampling,repair, eliminate_duplicates)

    def do(self, problem, n_samples, **kwargs):
        if self.pkl_data is None:
            self.sampling = problem.g.initialize_pop( int(np.ceil(n_samples * 1.5)) )
        else:
            self.sampling = load_pkl(self.pkl_data)

        if isinstance(self.sampling, np.ndarray):
            pop = Population.new(X=self.sampling)

        # filter duplicate in the population
        pop = self.eliminate_duplicates.do(pop)[:n_samples]

        if len(pop) < n_samples:
            raise ValueError('Initialization Class: Number of initial population is less then %d.'%n_samples)

        # print('pop = ', pop.get('X'),'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', len(pop))
        return pop
