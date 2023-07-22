# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import csv, operator
import pandas as pd
import pickle, traceback
import numpy as np

def csv_create_empty_df(file, cols):
    df = pd.DataFrame(data=None, columns=cols)
    df.to_csv('%s.csv'%file, index=False)

def csv_append_row(file, row):
    with open('%s.csv'%file, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def write_to_file(file, s):
    with open('%s _ Errors.txt'%file, 'a') as myfile:
        myfile.write(  s  )

def load_pkl(file_name):
    path = file_name if str(file_name).endswith('.pkl') else '%s.pkl'%file_name
    with open(path, "rb") as f:
        return pickle.load(f)

def dump(data, path):
    try:
        with open(path+'.pkl', "wb") as f:
            pickle.dump(data, f)
    except Exception as exc:
        print('\n**** Err:\n', traceback.format_exc())

def get_sorted_tuble_lst(tub_lst, descending=True, item=0):
    return sorted(tub_lst, key=operator.itemgetter(item), reverse=descending)

def rnd():
    return np.random.rand()