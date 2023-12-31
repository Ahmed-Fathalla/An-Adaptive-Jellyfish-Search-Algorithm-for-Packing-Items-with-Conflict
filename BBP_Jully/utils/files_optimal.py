# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

class_0 = [  "BPPC_1_1_0" ]
class_1_1 = [ 'BPPC_1_1_1', 'BPPC_1_2_5', 'BPPC_1_3_3',
              'BPPC_1_4_2', 'BPPC_1_5_5', 'BPPC_1_6_3']
class_1_2 = [ 'BPPC_1_7_2', 'BPPC_1_8_6', 'BPPC_1_9_4', 'BPPC_3_6_4']

class_2_1 = ['BPPC_2_3_5', 'BPPC_2_4_6', 'BPPC_2_5_7', 'BPPC_2_6_7', 'BPPC_2_7_5']
class_2_2 = ['BPPC_2_8_5', 'BPPC_2_9_2', 'BPPC_2_1_3', 'BPPC_2_2_2']

class_3_1 = ['BPPC_3_1_3', 'BPPC_3_2_1', 'BPPC_3_3_9']
class_3_2 = ['BPPC_3_4_5', 'BPPC_3_5_8']
class_3_3 = ['BPPC_3_7_5'] # -----------
class_3_4 = ['BPPC_3_8_3']
class_3_5 = ['BPPC_3_9_5']

problem = [class_1_1, class_1_2, class_2_1, class_2_2, class_3_1, class_3_2, class_3_3, class_3_4, class_3_5]

file_optimal={
                "BPPC_1_1_0": 2,
                "BPPC_1_1_1": 48,
                "BPPC_1_2_5": 50,
                "BPPC_1_3_3": 46,
                "BPPC_1_4_2": 53,
                "BPPC_1_5_5": 57,
                "BPPC_1_6_3": 75,
                "BPPC_1_7_2": 91,
                "BPPC_1_8_6": 99,
                "BPPC_1_9_4": 110,

                "BPPC_2_1_3": 102,
                "BPPC_2_2_2": 100,
                "BPPC_2_3_5": 101,
                "BPPC_2_4_6": 109,
                "BPPC_2_5_7": 126,
                "BPPC_2_6_7": 147,
                "BPPC_2_7_5": 181,
                "BPPC_2_8_5": 205,
                "BPPC_2_9_2": 225,

                "BPPC_3_1_3": 202,
                "BPPC_3_2_1": 198,
                "BPPC_3_3_9": 196,
                "BPPC_3_4_5": 206,
                "BPPC_3_5_8": 241,
                "BPPC_3_6_4": 307,
                "BPPC_3_7_5": 343,
                "BPPC_3_8_3": 400,
                "BPPC_3_9_5": 444,

                # "BPPC_1_1_1":48,
                "BPPC_1_1_2":49,
                "BPPC_1_1_3":46,
                "BPPC_1_1_4":49,
                "BPPC_1_1_5":50,
                "BPPC_1_1_6":48,
                "BPPC_1_1_7":48,
                "BPPC_1_1_8":49,
                "BPPC_1_1_9":50,
                "BPPC_1_1_10":46,
                "BPPC_1_2_1":48,
                "BPPC_1_2_2":49,
                "BPPC_1_2_3":46,
                "BPPC_1_2_4":49,
                # "BPPC_1_2_5":50,
                "BPPC_1_2_6":48,
                "BPPC_1_2_7":48,
                "BPPC_1_2_8":49,
                "BPPC_1_2_9":50,
                "BPPC_1_2_10":46

                 # # 'BPPC_1_0_3': 46,
                 # 'BPPC_1_1_1': 48,
                 # 'BPPC_1_2_5': 50,
                 # 'BPPC_1_3_3': 46,
                 # 'BPPC_1_4_2': 53,
                 # 'BPPC_1_5_5': 57,
                 # 'BPPC_1_6_3': 75,
                 # 'BPPC_1_7_2': 91,
                 # 'BPPC_1_8_6': 99,
                 # 'BPPC_1_9_4': 110,
                 #
                 # # 'BPPC_2_0_1': 99,
                 # 'BPPC_2_1_3': 102,
                 # 'BPPC_2_2_2': 100,
                 # 'BPPC_2_3_5': 101,
                 # 'BPPC_2_4_6': 109,
                 # 'BPPC_2_5_7': 126,
                 #
                 # 'BPPC_2_3_7': 102,
                 # 'BPPC_2_4_2': 105,
                 # 'BPPC_2_5_1': 111,
                 # 'BPPC_2_6_7': 147,
                 # 'BPPC_2_7_5': 181,
                 # 'BPPC_2_8_5': 205,
                 # 'BPPC_2_9_2': 225,
                 #
                 # # 'BPPC_3_0_6': 206,
                 # 'BPPC_3_1_3': 202,
                 # 'BPPC_3_2_1': 198,
                 # 'BPPC_3_3_9': 196,
                 # 'BPPC_3_4_5': 206,
                 # 'BPPC_3_5_8': 241,
                 #
                 # 'BPPC_3_4_10': 212,
                 # 'BPPC_3_5_4': 246,
                 # 'BPPC_3_6_4': 307,
                 # 'BPPC_3_7_5': 343,
                 # 'BPPC_3_8_3': 400,
                 # 'BPPC_3_9_5': 444
            }