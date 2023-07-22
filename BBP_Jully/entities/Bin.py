# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import numpy as np

class Bin:
    def __init__(self, id = 0, capacity = 150):
        self.id = id
        self.c = capacity
        self.remaining = capacity
        self.item_lst = []
        self.utility = 'Empty'

    def update_utility(self):
        self.utility = (self.c - self.remaining) ** 2 / self.c ** 2

    def pack_items(self, item_lst, item_dic):
        # item_lst_w = [item_dic[i].w for i in item_lst]
        # ** print('\n\tPacking... Bin:',self.id, 'Bin.items', self.item_lst , 'incomming:' ,item_lst , ' W:%r sum:%d'%(item_lst_w, sum(item_lst_w)))
        for i in item_lst:

            if self.remaining < item_dic[i].w:
                items_w = [item_dic[i].w for i in self.item_lst]
                raise ValueError(" Bin Class: Capacity  remaining:", self.remaining, " ,  bin_id", self.id, 'item_lst:', self.item_lst, 'W:', items_w,
                                 "trying to pack item_id:", i)

            other_items_in_bin = self.item_lst
            # other_items_in_bin.remove(i)
            #** print('\t\tpack_items -> Item:%d  Conf_lst:%r   Itms_in_lst:%r' %( i, item_dic[i].conflict_lst, other_items_in_bin))
            if len(np.intersect1d( item_dic[i].conflict_lst, self.item_lst )) > 0 :
                raise ValueError(" Bin Class:  Conflict,  bin_id", self.id, 'item_lst:', other_items_in_bin, "trying to pack item_id:", i, 'intersection:', np.intersect1d( item_dic[i].conflict_lst, self.item_lst ))

            self.item_lst.append(i)
            self.remaining -= item_dic[i].w

            #** print('\t\t', self.id, 'pack_items.remaining = ', self.remaining, '  items:', self.item_lst, ' W:', [item_dic[i].w for i in self.item_lst])
        self.utility = (self.c - self.remaining)**2 / self.c**2

    def pack_items_No_Check(self, item_lst, item_dic):
        for i in item_lst:
            self.item_lst.append(i)
            self.remaining -= item_dic[i].w
        self.utility = (self.c - self.remaining)**2 / self.c**2

    def pack_an_item(self, item_id, item_w):
        # print('Packing in Bin_%d item_%d  w_%d' % (self.id, item_id, item_w), '  Current bin items', self.item_lst)
        self.remaining -= item_w
        if self.remaining < 0:
            raise ValueError('Bin %d has a -ve remaining = %d'%(self.id, self.remaining))
        self.item_lst.append(item_id)

    def free_an_item(self, item_id, item_w):
        self.remaining += item_w
        self.item_lst.remove(item_id)

    def recheck_bin_remaining(self, item_dic):
        w, item_id = [], []
        for i in self.item_lst:
            w.append( item_dic[i].w )
            item_id.append( i )
        if self.remaining != self.c - np.array(w).sum():
            #
            raise ValueError('E001 not equal: Bin_id:%d'%self.id ,'   self.Remaining:%d'%self.remaining,  '    self.c-w:%d'%( self.c - np.array(w).sum()),
                             '  item_id:', item_id, ' W:', w, '=', np.array(w).sum()                              )

    def print_info(self, item_dic):
        self.recheck_bin_remaining(item_dic)
        w = [item_dic[i].w for i in self.item_lst]
        print('\t----- BinClass_B_id',self.id, ' items:', self.item_lst, ' W:',w, 'Sum:%d'%np.array(w).sum() , 'remaining:', self.remaining)