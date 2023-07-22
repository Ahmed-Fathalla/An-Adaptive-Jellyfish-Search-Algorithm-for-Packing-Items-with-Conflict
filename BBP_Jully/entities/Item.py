# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

class Item:
    def __init__(self, id=0, weight=0, conflict_lst=[]):
        self.id = id
        self.w = weight
        self.conflict_lst = conflict_lst
        self.packed_to_bin = -1
        self.color_id = '-'