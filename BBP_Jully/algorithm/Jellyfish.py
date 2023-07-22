# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import sys
from .Jullyfish_base import *

class Jellyfish(Jellyfish_base):
    def __init__(self,
                 pop_size=25,
                 trend_type = 'random',
                 eliminate_duplicates=True,
                 updating_method = 1,
                 update_Salah_No = 1,
                 **kwargs):
        self.cc = 0
        super().__init__(
                         updating_method=updating_method,
                         update_Salah_No = update_Salah_No,
                         # No_of_threads=No_of_threads,
                         trend_type = trend_type,
                         pop_size = pop_size,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs
                         )

    def interact_1(self, x):
        new_individual = copy.deepcopy(x)
        _, unique_bins, bin_utils, bins = self.dist_individual_bins_and_utils(new_individual)

        B_binID = bin_utils[-1][0]

        # items will be moved to A
        target_bins = list(set(unique_bins) - {B_binID})
        A_binID = np.random.choice( target_bins, 1, replace=False)[0]
        target_bins.remove(A_binID)

        # remove Number of self.update_Salah_No items from B to A
        successfully_moved_items = 0
        while True:
            for itm in bins[B_binID].item_lst:
                if bins[A_binID].remaining >= self.items[itm].w and len(np.intersect1d(bins[A_binID].item_lst, self.items[itm].conflict_lst)) == 0:
                    new_individual[itm - 1] = A_binID
                    bins[B_binID].free_an_item(itm, self.items[itm].w )
                    bins[A_binID].pack_an_item(itm, self.items[itm].w)
                    successfully_moved_items += 1

                if successfully_moved_items == self.update_Salah_No:
                    break

            if len(target_bins) == 0 or successfully_moved_items == self.update_Salah_No:
                break
            elif successfully_moved_items < self.update_Salah_No:
                A_binID = np.random.choice(target_bins, 1, replace=False)[0]
                target_bins.remove(A_binID)

        bins, new_individual = self.readjust_bins_order(bins, 'interact_1')
        if self.enable_trix:
            return self.extra_improve(new_individual)
        return new_individual

    def interact_2(self,a, b):
        a = np.array(a)
        b = np.array(b)
        new_individual = (np.ones( shape=len(a), dtype=int )*-100).tolist()

        fit_a = self.problem.g.fitness_value(a)
        fit_b = self.problem.g.fitness_value(b)

        if fit_a > fit_b:
            # b is better
            si, si_unique_bins, si_bin_utils, si_bins = self.dist_individual_bins_and_utils(b)
            sj, sj_unique_bins, sj_bin_utils, sj_bins = self.dist_individual_bins_and_utils(a)

        else:
            # a is better
            si, si_unique_bins, si_bin_utils, si_bins = self.dist_individual_bins_and_utils(a)
            sj, sj_unique_bins, sj_bin_utils, sj_bins = self.dist_individual_bins_and_utils(b)

        # s_i is better than s_j
        min_of_bins = min( len(si_unique_bins), len(sj_unique_bins) )
        N = np.random.randint(low=1, high= min_of_bins + 1 )

        new_individual_bins = copy.deepcopy(self.empty_bins)

        opened_bins = 1
        iteration = 0
        served_items = []
        while iteration < min_of_bins:
            for si_bin,_ in si_bin_utils[ N*iteration : N*(iteration+1) ]:
                new_lst = set(si_bins[si_bin].item_lst) - set(served_items)
                if len(new_lst)>0:
                    new_individual_bins[opened_bins].pack_items(new_lst, self.items)
                    served_items.extend( new_lst )
                    opened_bins += 1

            if len(served_items) == len(a):
                break

            new_lst = set(sj_bins[sj_bin_utils[iteration][0]].item_lst) - set(served_items)
            for item_ in new_lst: # iteration%len(sj_bin_utils)
                # print( 'for item_ = ' , item_ )
                success = False
                for bins_loop in range(1, opened_bins):
                    if new_individual_bins[bins_loop].remaining >= self.items[item_].w and \
                            len(np.intersect1d(new_individual_bins[bins_loop].item_lst, self.items[item_].conflict_lst)) == 0:
                        new_individual_bins[bins_loop].pack_an_item(item_, self.items[item_].w)
                        served_items.append( item_ )
                        success = True
                        break

                # Cant be served in the opened bins, so i will open a new bin
                if success==False:
                    new_individual_bins[opened_bins].pack_an_item(item_, self.items[item_].w)
                    served_items.append( item_ )
                    opened_bins += 1
            iteration += 1

            if len(served_items) == len(a):
                break

        rest_of_items = set(range(1,len(a)+1)) - set(served_items)
        for item_ in rest_of_items:
            success = False
            for bins_loop in range(1, opened_bins):
                if new_individual_bins[bins_loop].remaining >= self.items[item_].w and \
                        len(np.intersect1d(new_individual_bins[bins_loop].item_lst,
                                           self.items[item_].conflict_lst)) == 0:
                    new_individual_bins[bins_loop].pack_an_item(item_, self.items[item_].w)
                    served_items.append(item_)
                    success = True
                    opened_bins += 1
                    break

            # Cant be served in the opened bins, so i will open a new bin
            if success == False:
                new_individual_bins[opened_bins].pack_an_item(item_, self.items[item_].w)
                served_items.append(item_)
                opened_bins += 1

        if len(served_items) != len(a):
            raise ValueError('W001: Not all itemes are served', 'Seed:%d'%self.seed, 'gen_%d'%self.n_gen, self.problem_file)

        opened_bins -= 1

        new_individual_bins, new_individual = self.readjust_bins_order(new_individual_bins, 'interact_2')
        if self.enable_trix:
            return self.extra_improve(new_individual)
        return new_individual

    def extra_improve(self, x):
        x = np.array(x)
        ccc = 1
        _, unique_bins, bin_utils, bins = self.dist_individual_bins_and_utils(x)
        removing_bins_count = int(np.ceil(len(unique_bins) * self.extra_improving_ratio))

        sucess_bins = 0 # ========================================================================================================
        for count in range(removing_bins_count):
            removing_from_bin_id = bin_utils[-count-1][0]
            unique_bins.remove(removing_from_bin_id)
            removing_itms = bins[removing_from_bin_id].item_lst
            for itm in removing_itms:
                for other_bin in unique_bins:
                    if bins[other_bin].remaining >= self.items[itm].w and \
                            len(np.intersect1d(bins[other_bin].item_lst,
                                               self.items[itm].conflict_lst)) == 0:
                        bins[removing_from_bin_id].free_an_item(itm, self.items[itm].w)
                        bins[other_bin].pack_an_item(itm, self.items[itm].w)
                        break

        new_individual_bins, new_individual = self.readjust_bins_order(bins)
        return new_individual

    def minus_(self, a, b):
        index_of_non_similar_cols = list(np.where( (a - b)!=0 )[0] )
        lst = []
        for i in index_of_non_similar_cols:
            lst.append( [ i+1, a[i], b[i] ]  )
        return lst

    def plus_(self, x, S):
        _, bins_, items_dict = self.dist_individual(x, 'C', get_items_dic=True)
        new_individual = copy.deepcopy(x)
        sucess_swaps = 0
        for s in S:
            item_ = int(s[0])
            from_ = int(items_dict[item_].packed_to_bin )  # from_ = s[1]
            to_ = int(s[2])


            if from_ == to_:
                continue

            if bins_[ to_ ].remaining >= items_dict[ item_ ].w and len(np.intersect1d( bins_[ to_ ].item_lst, items_dict[ item_ ].conflict_lst)) == 0:

                bins_[from_].free_an_item( item_ , items_dict[ item_ ].w )
                bins_[to_].pack_an_item( item_, items_dict[item_].w)
                items_dict[item_].packed_to_bin = to_
                sucess_swaps += 1
            elif self.enable_trix:
                unique_bins = list(set(x))
                unique_bins.remove(from_)
                unique_bins.remove(to_)
                found = False
                for bin_id in unique_bins:
                    if bins_[bin_id].remaining >= items_dict[item_].w and len(np.intersect1d(bins_[bin_id].item_lst, items_dict[item_].conflict_lst)) == 0:
                        bins_[from_].free_an_item(item_, items_dict[item_].w)
                        bins_[bin_id].pack_an_item(item_, items_dict[item_].w)
                        items_dict[item_].packed_to_bin = bin_id
                        sucess_swaps += 1
                        found = True
                        break
                if not found:
                    ...

        # ===================
        # readjust_bins_order
        # ===================
        bins_, new_individual = self.readjust_bins_order(bins_)
        if self.enable_trix:
            return self.extra_improve(new_individual)
        return new_individual

    def dot_(self, a, S):
        len_ = int(np.ceil(a * len(S)))
        indices = np.random.choice(list(range(0, len(S))), len_, replace=False)
        output = np.array(S)[ np.array(indices, dtype=int)]
        return output

