# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import random
from ..utils.utils_ import get_sorted_tuble_lst
from .Graph_base import *

def get_data_from_file(file_path):
    ...

class Graph(Graph_base):
    def __init__(self, file, initialization_method=None):
        super().__init__(file=file, initialization_method=initialization_method)

    def initialize_pop(self, pop_size):
        pop = None

        if self.initialization_method == 'random_First_Fit':
            pop = self.Inintialization_random_first(pop_size)
        elif self.initialization_method == 'greedy':
            pop = self.Initialization_greedy(pop_size)
        elif self.initialization_method == 'random_random':
            pop = self.Inintialization_random_random(pop_size)
        return pop

    def Initialization_greedy(self, pop_size):
        pop = []
        for i in range(pop_size):
            pop.append(self.greedy_coloring_algorithm(get_individual=True, seed=i))
        return np.array(pop)

    def first_fit(self):
        bins = copy.deepcopy(self.empty_bins)

        itms = list(range(1, len(self.items) + 1))

        for itm in itms:
            i = 1
            while True:
                if bins[i].remaining < self.items[itm].w:
                    i += 1
                    continue
                if len(np.intersect1d(bins[i].item_lst, self.items[itm].conflict_lst)) > 0:
                    i += 1
                    continue
                else:
                    bins[i].item_lst.append(itm)
                    self.items[itm].packed_to_bin = i
                    bins[i].remaining -= self.items[itm].w
                    break

        individual = [self.items[k].packed_to_bin for k in sorted(self.items.keys())]
        fitness, bins = self.fitness_value( individual )

        return fitness, bins, individual

    def best_fit(self):
        bins = copy.deepcopy(self.empty_bins)
        itms = list(range(1, len(self.items) + 1))

        max_bin_id = 1

        for itm in itms:
            options = []

            for i in range(1, max_bin_id+1):
                if bins[i].remaining < self.items[itm].w:
                    i += 1
                    continue
                if len(np.intersect1d(bins[i].item_lst, self.items[itm].conflict_lst)) > 0:
                    i += 1
                    continue
                else:
                    Utility_if_is_packed = (bins[i].c - (bins[i].remaining - self.items[itm].w)  ) ** 2 / bins[i].c ** 2
                    options.append( (i, Utility_if_is_packed) )

            i = max_bin_id+1
            Utility_if_is_packed = (bins[i].c - (bins[i].remaining - self.items[itm].w)) ** 2 / bins[i].c ** 2
            options.append(( i , Utility_if_is_packed))

            sorted_lst = get_sorted_tuble_lst(options, item=1)
            best_bin_id = sorted_lst[0][0]

            # pack the item
            # =============
            if best_bin_id == max_bin_id + 1:
                max_bin_id += 1

            self.items[itm].packed_to_bin = best_bin_id
            bins[best_bin_id].pack_an_item(itm, self.items[itm].w)

        individual = [self.items[k].packed_to_bin for k in sorted(self.items.keys())]
        fitness, bins = self.fitness_value( individual )
        return fitness, bins, individual

    def greedy_fit(self):
        self.greedy_coloring_algorithm(shuffle_=False)
        self.distribute_conf_items_on_bins()
        self.distribute_non_conf_items_on_bins()
        return self.get_individual()

    def Inintialization_random_random(self, pop_size=1):
        pop = []
        for _ in range(pop_size):
            bins = copy.deepcopy(self.empty_bins)

            itms = list(range(1, len(self.items) + 1))
            random.shuffle(itms)  # step 1 random ordering
            opened_bins = [1]
            for itm in itms:
                chosen_bin = random.choice(opened_bins)

                if bins[chosen_bin].remaining > self.items[itm].w and \
                        len(np.intersect1d(bins[chosen_bin].item_lst, self.items[itm].conflict_lst)) == 0:
                    bins[chosen_bin].pack_an_item(itm, self.items[itm].w)
                    self.items[itm].packed_to_bin = chosen_bin
                else:
                    chosen_bin = len(opened_bins)+1
                    opened_bins.append( chosen_bin )

                    bins[chosen_bin].pack_an_item(itm, self.items[itm].w)
                    self.items[itm].packed_to_bin = chosen_bin
            pop.append([ self.items[k].packed_to_bin
                         for k in sorted(self.items.keys())
                        ])
        return np.array(pop)

    def Inintialization_random_first(self, pop_size=1):
        pop = []
        for _ in range(pop_size):
            bins = copy.deepcopy(self.empty_bins)

            itms = list(range(1, len(self.items) + 1))
            random.shuffle(itms)  # step 1 random ordering

            for itm in itms:
                i = 1
                while True:
                    if not bins[i].remaining > self.items[itm].w:
                        i += 1
                        continue
                    if len(np.intersect1d(bins[i].item_lst, self.items[itm].conflict_lst)) > 0:
                        i += 1
                        continue
                    else:
                        bins[i].item_lst.append(itm)
                        self.items[itm].packed_to_bin = i
                        bins[i].remaining -= self.items[itm].w
                        # print('      ', node, 'bin:', i)
                        break

            pop.append([self.items[k].packed_to_bin for k in sorted(self.items.keys())])
        return np.array(pop)

