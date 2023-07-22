# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

from pymoo.core.termination import Termination
import numpy as np

class Fathalla_termination(Termination):
    def __init__(self, n_max_gen, plateau_threshold) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

        if self.n_max_gen is None:
            self.n_max_gen = float("inf")

    def do_continue(self, algorithm):
        return algorithm.n_gen < self.n_max_gen and \
               len(set(algorithm.opt.get('X')[0])) > algorithm.Optimal_solution