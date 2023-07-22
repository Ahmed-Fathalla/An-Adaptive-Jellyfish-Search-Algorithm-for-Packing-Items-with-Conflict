# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

import matplotlib.pyplot as plt
import pandas as pd
import signal

def plt_(file, fontsize=20, save_plt=True, mark_size=8):
    symbol = ["r.-", "b*--", "go-.", "c^-", "mx-" ]

    df = pd.read_csv('%s.csv'%file)
    runs = df['Run_id'].unique().tolist()
    plt.rcParams.update({'figure.figsize':(20,15), 'figure.dpi':120})

    fig, axes = plt.subplots(2, 1)

    axes[0].set_title('Optimal Fitness', fontsize=30)


    for index_, i in enumerate(runs):
        dd = df[ df['Run_id']==i ].copy()

        lst = dd['Fitness'].values
        axes[0].plot( range(1, len(lst)+1) , lst, symbol[index_-1], label='Run_ID %d'%index_, markersize=mark_size)

        lst = dd['Bins'].values
        axes[1].plot( range(1, len(lst)+1) , lst, symbol[index_-1], label='Run_ID %d'%index_, markersize=mark_size)

    axes[0].legend(fontsize=12)
    axes[0].set_xlabel('Iterations', fontsize=fontsize)
    axes[0].set_ylabel('Fitness', fontsize=fontsize)
    axes[0].grid()

    axes[1].set_title('Optimal Bins', fontsize=30)
    axes[1].legend(fontsize=12)
    axes[1].set_xlabel('Iterations', fontsize=fontsize)
    axes[1].set_ylabel('No of Bins', fontsize=fontsize)

    plt.grid()
    plt.subplots_adjust(
                        left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    if save_plt:plt.savefig('%s.pdf'%file, bbox_inches='tight')
    plt.show()