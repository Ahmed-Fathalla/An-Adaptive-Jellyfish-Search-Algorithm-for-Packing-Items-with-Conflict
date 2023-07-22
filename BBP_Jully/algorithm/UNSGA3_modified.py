# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.misc import intersect, has_feasible

from ..utils.utils_ import csv_append_row
import numpy as np

class UNSGA3_modified(UNSGA3):
    def __init__(self,
                 ref_dirs,
                 iteration,
                 output_per_gen_file,
                 **kwargs):
        self.iteration = iteration
        self.output_per_gen_file = output_per_gen_file
        self.Optimal_solution = None
        self.problem_file = None
        super().__init__(ref_dirs = ref_dirs,
                         **kwargs)

    def _set_optimum(self, **kwargs):
        super()._set_optimum(**kwargs)
        csv_append_row(self.output_per_gen_file, [self.iteration, self.n_gen, self.opt.get('F')[0][0], len(set(self.opt.get('X')[0]))])

    def _infill(self):
        print("\r", 'GA ' + self.problem_file, '%s/%-4d  Best (bins:%d(%d), fitness:%-.5f)' % (str(self.iteration),
                                                                                       self.n_gen,
                                                                                       len(set(self.opt.get('X')[0])),
                                                                                       self.Optimal_solution,
                                                                                       self.opt.get('F')[0][0]),
                                                                                       end="")
        return super()._infill()

    def _initialize_infill(self):
        self.problem_file = self.problem.g.problem_file.split('/')[-1][:-4]
        self.Optimal_solution = self.problem.optimal_No_of_bins
        return super()._initialize_infill()


