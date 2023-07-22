# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import numpy as np
import pandas as pd
import copy, os, gc
from ..entities.Graph import Graph
import time
import sys
from pymoo.core.problem import ElementwiseProblem
class BPP_wz_conlict(ElementwiseProblem):
    def __init__(self,
                 problem_file,
                 optimal_No_of_bins,
                 # Max_iteration,
                 np_seed=None,
                 exp_id = '',
                 initialization_method=None,


                 **kwargs):
        if np_seed is not None:np.random.seed(np_seed)
        self.optimal_No_of_bins = optimal_No_of_bins

        self.g = Graph(problem_file, initialization_method)
        self.i = 0
        self.exp_id = exp_id
        super().__init__(
            n_var=len(self.g.items),
            n_obj=1,
            xl=0,
            xu = len(self.g.bins),
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        self.i += 1
        out['F'], No_of_unique_bins = self.g.fitness_value(x)