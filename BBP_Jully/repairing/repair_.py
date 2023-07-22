# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import numpy as np
from pymoo.core.repair import Repair
from ..algorithm.base import *

class MyRepair(Repair, Base_):
    def __init__(self, problem, seed, exp_str=''):
        super(MyRepair, self).__init__()

        self.problem = problem
        self.problem_file = self.problem.g.problem_file.split('/')[-1][:-4]
        self.bin_capacity = self.problem.g.bin_capacity
        self.items = self.problem.g.items
        self.item_dict_empty_bins = self.problem.g.item_dict_empty_bins

        self.empty_bins = copy.deepcopy(self.problem.g.empty_bins)
        self.item_max = np.max(list(self.items.keys()))
        self.seed = seed

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        for k in range(len(X)):
            xxxx = self.doo(X[k])
            pop[k].set("X", xxxx)
        return pop

    def doo(self, x=None, return_=False):
        xxxx = x.copy()
        xxxx = np.where(xxxx == 0, 1, xxxx)

        if xxxx.max() != len(set(xxxx.tolist())):
            set_ = set(xxxx.tolist())
            sorted(set_)
            max_ = max(set_)
            enu = 1
            for lst_itm in set_:
                while lst_itm != enu:
                    xxxx = np.where(xxxx == max_, enu, xxxx)
                    enu += 1
                    if enu > max_:
                        break
                enu += 1

                set_ = set(xxxx.tolist())
                sorted(set_)


        conf_items = []
        unique_bins, bins = self.dist_individual_no_check(xxxx)

        # ==============================================================================================================
        # Collect "conf_items"
        for bin_id in unique_bins:

            items = bins[bin_id].item_lst
            items_cpd = items.copy()

            # remove conf items
            for itm in items:
                ww = np.intersect1d( list( set(items_cpd)-{itm} ) ,  self.items[itm].conflict_lst     )
                if len( ww ) > 0:
                    bins[bin_id].free_an_item(itm, self.items[itm].w)
                    items_cpd.remove(itm)
                    conf_items.append(itm)

            # Check capacity
            while bins[bin_id].remaining < 0:
                itm = items_cpd[0]

                bins[bin_id].free_an_item(itm, self.items[itm].w)
                items_cpd.remove(itm)
                conf_items.append(itm)

        ####################
        unique_bins = [b for b in bins.keys() if len(bins[b].item_lst) > 0]
        # print('non_empty_bins = ', unique_bins)
        empty_bins = [i for i in bins.keys() if i not in unique_bins]
        # print('empty_bins = ', empty_bins)
        ####################


        # ==============================================================================================================
        # pack "conf_items" items
        for itm in conf_items:
            found = False




            for bin_id in unique_bins:
                if bins[bin_id].remaining >= self.items[itm].w and \
                        len( np.intersect1d(bins[bin_id].item_lst, self.items[itm].conflict_lst)) == 0:
                    bins[bin_id].pack_an_item(itm, self.items[itm].w) #########################
                    found = True
                    break

            if found == False:
                new_bin = empty_bins.pop(0)
                unique_bins.append( new_bin )
                bins[new_bin].pack_an_item(itm, self.items[itm].w)

        _, new_individual = self.readjust_bins_order(bins)
        return new_individual