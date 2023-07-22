# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import copy
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder as le
import pandas as pd
import random
import matplotlib.pyplot as plt
from ..entities.Bin import Bin
from ..entities.Item import Item
from ..utils.time_utils import get_TimeStamp_str
from ..utils.colors import color_lst

# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
# https://towardsdatascience.com/graph-coloring-with-networkx-88c45f09b8f4   "greedy_coloring_algorithm" implementation
class Graph_base:
    def __init__(self, file, initialization_method):

        self.problem_file = file
        self.initialization_method = initialization_method

        self.colors = color_lst
        self.conf_items = []
        self.non_conf_items = []
        self.color_items = {}

        self.items = {}
        self.bins = {}
        with open(self.problem_file, 'r') as myfile:
            z = myfile.read().split('\n')

        k = [int(j) for j in z[0].split()]
        for i in z[1:]:
            if i.strip() == '':
                continue
            i = [int(j) for j in i.split()]
            self.items[i[0]] = Item(id=i[0], weight=i[1], conflict_lst=i[2:])

            self.conf_items.extend( i[2:] )

            if len(self.items[i[0]].conflict_lst) > 0:
                self.conf_items.append(i[0])



        self.no_of_items = len(self.items)
        self.no_of_bins = k[0]
        self.bin_capacity = k[1]

        self.network = nx.Graph()
        self.edges = [[min(itm, cnf_itm), max(itm, cnf_itm)] for itm in self.conf_items for cnf_itm in
                      self.items[itm].conflict_lst]
        self.network.add_nodes_from(list(set(np.array(self.edges).flatten().tolist())))
        for i in self.edges:
            self.network.add_edge(*i)

        for conf_item in self.conf_items:
            self.items[conf_item].conflict_lst = list(self.network[conf_item])

        self.item_dict_empty_bins = copy.deepcopy(self.items)

        self.non_conf_items = self.items.keys() - list(self.network.nodes)
        self.max_deg = max( list(np.array( self.network.degree )[:, 1]) )
        self.bins = self.build_bins()
        self.empty_bins = copy.deepcopy(self.bins)

        self.set_items_w()
        self.conf_items = list(set(self.conf_items))

    def Inintialization_random_wz_constraints(self, pop_size=1):
        ...

    def initialize_pop(self, pop_size):
        ...

    def show_bin_items(self, bins_obj=None):
        if bins_obj is not None:self.bins=copy.deepcopy(bins_obj)
        s = ''
        total_bin_items_w = 0
        for _, bin in self.bins.items():
            if len(bin.item_lst) == 0:
                continue
            s += 'Bin:%-3d ' % bin.id + ' remaining:%-3d   ' % bin.remaining
            bin_item_weights = 0
            for item in bin.item_lst:
                s += '(item:%d(color:%s)-w:%d), ' % (self.items[item].id, self.items[item].color_id, self.items[item].w)
                bin_item_weights += self.items[item].w

            total_bin_items_w += bin_item_weights ** 2
            s = s[:-2] + '\n'

        self.no_of_used_bins = len(s.split('\n')) - 1

        denomerator = self.no_of_used_bins * self.bin_capacity ** 2
        fitness = 1.0 - total_bin_items_w / denomerator
        print(s + '\n\n No_of_bins:' + str(self.no_of_used_bins) + '  Max_deg: %d' % self.max_deg, ' fitness:', fitness)

    def fitness_value(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        total_bin_items_w = 0

        unique_bins = list(set(x))
        for bin_id in unique_bins:
            items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]
            bin_item_weights = 0
            for item in items_in_bin:
                bin_item_weights += self.items[item].w
            total_bin_items_w += bin_item_weights ** 2

        denomerator = len(unique_bins) * self.bin_capacity ** 2
        fitness = 1.0 - total_bin_items_w / denomerator

        return fitness, len(unique_bins)

    def fitness_value_trace(self, x):
        s = ''
        x = np.array(x)
        total_bin_items_w = 0
        unique_bins = list(set(x))
        print('unique_bins = ', unique_bins)
        for bin_id in unique_bins:
            items_in_bin = [j + 1 for j in list(np.where(x == bin_id)[0])]
            bin_item_weights = 0
            s = '  Items.w:'
            for item in items_in_bin:
                s += ' %d, ' % self.items[item].w
                bin_item_weights += self.items[item].w

            s = s[:-2] + '   fill_k^2: %d' % bin_item_weights ** 2
            total_bin_items_w += bin_item_weights ** 2

        denomerator = len(unique_bins) * self.bin_capacity ** 2
        fitness = 1.0 - total_bin_items_w / denomerator

        return fitness, len(unique_bins)

    def set_items_w(self):
        for node in list(self.network.nodes()):
            self.network.nodes[node]['w'] = self.items[node].w

    def build_bins(self):
        bins = {}
        for i in range(1, self.no_of_bins + 1):
            bins[i] = Bin(id=i, capacity=self.bin_capacity)
        return bins

    def get_neihbors(self, node):
        return sorted([x for x in self.network.neighbors(node)])

    # ========================================================================================================================================================================
    def get_individual(self):
        encoder = le()
        encoder.fit(np.array([self.network.nodes[node]['color'] for node in self.conf_items]))
        return [encoder.transform(np.array(self.network.nodes[node]['color']).reshape(-1, 1))[0] for node in
                self.conf_items]

    def Initialization_greedy(self, pop_size):
        ...

    def greedy_coloring_algorithm(self, print_color_items=False, shuffle_=True, get_individual=False, seed=1):
        # https://towardsdatascience.com/graph-coloring-with-networkx-88c45f09b8f4
        nodes = list(self.network.nodes())

        self.set_color_dic()
        for node in list(self.network.nodes()):
            if 'color' in self.network.nodes[node].keys():
                del self.network.nodes[node]['color']

        random.seed(seed)
        if shuffle_:
            random.shuffle(nodes)  # step 1 random ordering
            random.shuffle(self.colors)

        selected_colors = []
        color_itm_dict = {}
        for node in nodes:
            # print('node = ', node)
            dict_neighbors = dict(self.network[node])  # gives names of nodes that are neighbors
            nodes_neighbors = list(dict_neighbors.keys())
            forbidden_colors = []
            for neighbor in nodes_neighbors:
                if 'color' not in list(self.network.nodes.data()[neighbor].keys()):
                    # if the neighbor has no color, proceed
                    continue
                else:
                    forbidden_color = self.network.nodes.data()[neighbor]
                    forbidden_color = forbidden_color['color']
                    forbidden_colors.append(forbidden_color)  # assign the first color


            for color in self.colors:
                if color in forbidden_colors:
                    continue
                elif self.color_dic[color] < self.items[node].w:
                    continue
                else:

                    if color in color_itm_dict.keys():
                        color_itm_dict[color].append(node)
                    else:
                        color_itm_dict[color] = [node]

                    selected_colors.append( color )
                    self.network.nodes[node]['color'] = color
                    self.color_dic[color] -= self.items[node].w
                    break


        for i,color in enumerate( list(set(selected_colors)),1):
            self.bins[i].pack_items(color_itm_dict[color], self.items)

        self.get_color_items(print_color_items=print_color_items)
        if get_individual:
            return self.get_individual()

    def get_color_items(self, print_color_items=False):
        self.color_items = {}
        for node_id, v in self.network.nodes.data():
            if v['color'] not in self.color_items.keys():
                self.color_items[v['color']] = [node_id]
            else:
                self.color_items[v['color']].append(node_id)
        if print_color_items:
            for k, v in self.color_items.items():
                print(k, ": ", v, sep='')
            print('No of colors:', len(self.color_items.keys()))

    def distribute_conf_items_on_bins(self):
        bin_id = 0
        for iteration, (k, v) in enumerate(self.color_items.items(), 1):
            bin_id += 1
            for item_number in v:
                self.items[item_number].color_id = str(iteration)
                if self.items[item_number].w > self.bins[bin_id].remaining:
                    print('_-_-_-_-_-> 1')
                    bin_id += 1
                self.bins[bin_id].item_lst.append(item_number)
                self.bins[bin_id].remaining -= self.items[item_number].w

    def distribute_non_conf_items_on_bins(self):
        for non_conf_item in self.non_conf_items:
            bin_id = 1
            while True:
                if self.items[non_conf_item].w <= self.bins[bin_id].remaining:
                    self.bins[bin_id].pack_an_item( non_conf_item, self.items[non_conf_item].w  )
                    break
                bin_id += 1

    def set_color_dic(self):
        self.color_dic = {}
        for color in self.colors:
            self.color_dic[color] = self.bin_capacity