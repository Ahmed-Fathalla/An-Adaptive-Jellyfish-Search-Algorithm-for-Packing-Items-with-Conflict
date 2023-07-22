# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import copy, gc, traceback
import numpy as np
import pandas as pd
from pymoo.core.algorithm import Algorithm, filter_optimum
from pymoo.core.population import Population
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.evaluator import Evaluator
from ..entities.Bin import Bin
from ..utils.utils_ import *
from pymoo.core.population import Population
import time
import sys

class Base_:
    def __init__(self):
        super(Base_, self).__init__()
        self.problem = None
        self.problem_file = None
        self.bin_capacity = None
        self.items = None
        self.item_dict_empty_bins = None

        self.empty_bins = None
        self.item_max = None
        self.seed_ = None
        self.n_gen = None
        # print('Base_ = <<' )

    def check_conf(self, x, habal):
        _, bins = self.dist_individual(x, str(habal) + ' check_conf')
        unique_bins = list(set(x))
        for bin_id in unique_bins:
            items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]
            w = np.array([self.items[i].w for i in items_in_bin])

            for item in items_in_bin:
                other_items_on_bin = list(set(items_in_bin) - {item})
                conf_lst_len = len(np.intersect1d(self.items[item].conflict_lst, other_items_on_bin))
                if conf_lst_len > 0:
                    raise ValueError(
                        " Jullyfish Class: B002_seed_%d: %s  Confilict error, No_of_conflicts:" % (self.seed_, habal),
                        conf_lst_len,
                        '\nBin_id:', bin_id, ' Item_id ', item,
                        'conflict_lst:', self.items[item].conflict_lst, 'items_on_bin', other_items_on_bin,
                        'Seed:%d' % self.seed_, 'gen_%d' % self.n_gen,
                        self.problem_file, ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))

            if self.problem.g.bin_capacity - w.sum() != bins[bin_id].remaining:
                item_w = [self.items[i].w for i in items_in_bin]
                bins[bin_id].recheck_bin_remaining(self.items)
                raise ValueError(" Jullyfishhabal  Capacity error  Bin_id:" % (self.seed_, habal), bin_id,
                                 ' items_in_bin',
                                 items_in_bin, 'W:', item_w,
                                 '  remaining:', bins[bin_id].remaining, 'Seed:%d' % self.seed_, 'gen_%d' % self.n_gen,
                                 self.problem_file,
                                 ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))

        # Final Check
        self.print_bins_items(bins, print_=False)

    def print_bins_items(self, bins, print_=True):
        lst = []
        if print_: print('\n' * 2)
        for bid in bins.keys():
            if len(bins[bid].item_lst) > 0:
                bins[bid].update_utility()
                w = self.get_w(bins[bid].item_lst)
                if print_: print('Bin_id: %d' % bid, bins[bid].item_lst, 'Fitness_%-.3f' % bins[bid].utility, ' W:', w,
                                 'Sum_W:%d' % sum(w), 'Remain:', bins[bid].remaining)
                lst.append(bid)

        if len(lst) == 0:
            print('new_individual_bins.keys() = ', bins.keys())
            for bid in bins.keys():
                print(bid, bins[bid].item_lst)
            raise ValueError('X001: keys() == 0', 'Seed:%d' % self.seed_, 'gen_%d' % self.n_gen, self.problem_file,
                             ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))
        elif max(lst) != len(lst):
            # assert False, 'X002: Len and max dont mathch max:%d len:%d'%(max(lst), len(lst))
            raise ValueError('X002: Len and max dont mathch max:%d len:%d' % (max(lst), len(lst)),
                             'Seed:%d' % self.seed_, 'gen_%d' % self.n_gen, self.problem_file,
                             ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))

    def get_w(self, lst):
        return [self.items[itm].w for itm in lst]

    def calc_utility(self, bin_item_lst):
        w = 0
        for item in bin_item_lst:
            w += self.items[item].w
        return w ** 2 / self.problem.g.bin_capacity ** 2

    def print_pop(self, X, F, str_):
        for x, f in zip(X, F):
            ...

    def dist_individual_no_check(self, x, habal='', get_items_dic=False):
        item_dict = copy.deepcopy(self.item_dict_empty_bins)

        bins_ = copy.deepcopy(self.empty_bins)
        unique_bins = []
        try:
            unique_bins = list(set(x))
            for bin_id in unique_bins:
                items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]

                bins_[bin_id].pack_items_No_Check(items_in_bin, item_dict)
                if get_items_dic:
                    for item_id in items_in_bin:
                        item_dict[item_id].packed_to_bin = bin_id
        except Exception as exc:
            print('\n**** Err: J0012: \n', traceback.format_exc(), ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))
            raise ValueError('\n**** Err: J0012: \n', traceback.format_exc(),
                             ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))
        if get_items_dic:
            return unique_bins, bins_, item_dict
        return unique_bins, bins_

    def dist_individual(self, x, habal='', get_items_dic = False):
        item_dict = copy.deepcopy(self.item_dict_empty_bins)

        bins_ = copy.deepcopy(self.empty_bins)
        for _,b in bins_.items():
            if b.remaining != self.problem.g.bin_capacity:
                raise ValueError('J0010 Bin is not Empty', 'Seed:%d'%self.seed_, 'gen_%d'%self.n_gen, self.problem_file)

        unique_bins = []
        try:
            unique_bins = list(set(x))

            if max(unique_bins) != len(unique_bins):
                print('habal = ', habal)
                raise ValueError('x_1:', x, '\n**** Err: J0011: max:%d len:%d \n'%(max(unique_bins),len(unique_bins)),
                                 ' Gen_%d  seed_%d '%(self.n_gen,self.seed_),
                                 '^^^^^^^^^^^^ habal:',habal,' dist_individual ===== x', x)

            for bin_id in unique_bins:
                items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]
                bins_[bin_id].pack_items(items_in_bin, item_dict)
                if get_items_dic:
                    for item_id in items_in_bin:
                        item_dict[item_id].packed_to_bin = bin_id
        except Exception as exc:
            print('\n**** Err: J0012: \n', traceback.format_exc(), ' Gen_%d  seed_%d '%(self.n_gen,self.seed_))
            raise ValueError('\n**** Err: J0012: \n', traceback.format_exc(), ' Gen_%d  seed_%d '%(self.n_gen,self.seed_))

        if get_items_dic:
            return unique_bins, bins_, item_dict
        return unique_bins, bins_

    def bin_to_individual(self, bins):
        non_empty_bins = [b for b in bins.keys() if len(bins[b].item_lst) > 0]
        new_individual = (np.ones(shape=len(self.items.keys()), dtype=int) * -100).tolist()
        for bin_id in non_empty_bins:
            for itm_ in bins[bin_id].item_lst:
                new_individual[itm_ - 1] = bin_id
        return new_individual


    def readjust_bins_order(self, bins, habal=''):
        non_empty_bins = [b for b in bins.keys() if len(bins[b].item_lst) > 0]

        max_ = max(non_empty_bins)
        enu = 1
        for lst_itm in non_empty_bins:
            while lst_itm != enu:
                bins[enu] = copy.deepcopy(bins[max_])
                bins[max_] = Bin()
                non_empty_bins.remove(max_)
                max_ = max(non_empty_bins)
                enu += 1
                if enu > max_:
                    break
            enu += 1


        new_individual = (np.ones(shape=len(self.items.keys()), dtype=int) * -100).tolist()

        non_empty_bins = [b for b in bins.keys() if len(bins[b].item_lst) > 0]
        bin_id = -1
        try:
            for bin_id in non_empty_bins:
                # print('bin_id = ', bin_id)
                for itm_ in bins[bin_id].item_lst:
                    new_individual[itm_ - 1] = bin_id
        except Exception as exc:
            print('=====> new_individual_bins[%d].item_lst' % bin_id, 'Opened_Bins:',
                  [i for i in bins.keys() if len(bins[i].item_lst) > 0])
            self.print_bins_items(bins)
            print('\n**** Err:\n', traceback.format_exc())
            raise ValueError('Y002: Error', 'Seed:%d' % self.seed_, 'gen_%d' % self.n_gen, self.problem_file)

        # ========================================
        ### Checking 2
        # ========================================
        unique_bins = list(set(new_individual))
        if max(unique_bins) != len(unique_bins):
            print('x_2:', new_individual)
            raise ValueError('x_22:', new_individual,
                             '\n**** Err: KB001: max:%d len:%d \n' % (max(unique_bins), len(unique_bins)),
                             ' Gen_%d  seed_%d ' % (self.n_gen, self.seed_))

        # ===========================================
        ### Checking 3
        # ========================================
        if len(np.where(np.array(new_individual) < 0)[0].tolist()) > 0:
            print('\n\nindividual = ', new_individual)
            print('np.where( -ve values ) = ', np.where(np.array(new_individual) < 0)[0].tolist())
            raise ValueError('W0055_-ve is found in Unique bins', 'Seed:%d' % self.seed_,
                             'gen_%d' % self.n_gen, self.problem_file)

        return bins, new_individual

    def dist_individual_bins_and_utils(self, x, habal=''):
        tmp_bins = copy.deepcopy(self.empty_bins)
        for _,b in tmp_bins.items():
            if b.remaining != self.bin_capacity:
                raise ValueError('D002 Bin is not Empty', 'Seed:%d'%self.seed, 'gen_%d'%self.n_gen, self.problem_file, ' Gen_%d  seed_%d '%(self.n_gen,self.seed))

        unique_bins = []
        bins_and_utils = []
        try:
            unique_bins = list(set(x))
            for bin_id in unique_bins:
                items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]
                tmp_bins[bin_id].pack_items(items_in_bin, self.items)
                w = 0
                for item_id in items_in_bin:
                    self.items[item_id].packed_to_bin = bin_id
                    w += self.items[item_id].w

                bins_and_utils.append( [ bin_id, w**2 / self.problem.g.bin_capacity**2 ] )

        except Exception as exc:
            print('\n**** Err: K001:  Gen_%d \n'%self.n_gen, traceback.format_exc(), ' Gen_%d  seed_%d '%(self.n_gen,self.seed))

        return x, unique_bins, get_sorted_tuble_lst(bins_and_utils, item=1, descending=True), tmp_bins