# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief: 
"""

import sys

import numpy as np
import time, os, pickle, traceback

from .algorithm.Jellyfish import Jellyfish
from .algorithm.Jaya import Jaya
from .algorithm.PSO import PSO
from .algorithm.UNSGA3_modified import UNSGA3_modified

from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from .utils.time_utils import get_TimeStamp_str
from .utils.files_optimal import file_optimal
from .initialization.initialization import Fathalla_initialization
from .termination.termination import Fathalla_termination
from .problem.problem import BPP_wz_conlict
from .repairing.repair_ import MyRepair
from .utils.plt_convergence import plt_
from .utils.utils_ import *

# import multiprocessing
# from pymoo.core.problem import starmap_parallelized_eval

def load_exp(exp_path):
    return Exp().load(path=exp_path)

init_method_dic = { 'random_First_Fit': 'rf', 'random_random':'rr'}

class Exp:
    def __init__(self, file=None, algorithm_type=None, initialization_method=None,
                 pop_size=None, updating_method=None, Max_iteration=None, rand=None,
                 update_Salah_No = 1, plateau_threshold=6,
                 No_of_threads=5,
                 plt_convergence=True, save_initial_pop=False,
                 dump=False, start_extra_improving=300, save_history = False,
                 n_offsprings=30, n_partitions=12):

        if file is None:
            return

        self.file = file
        self.optimal_No_of_bins = file_optimal[file]
        self.initialization_method = initialization_method
        self.pop_size = pop_size
        self.updating_method = updating_method
        self.update_Salah_No = update_Salah_No

        if Max_iteration is not None:
            self.Max_iteration = Max_iteration
        else:
            if 'BPPC_1_' in self.file:self.Max_iteration = 800
            elif 'BPPC_2_' in self.file:self.Max_iteration = 1500
            elif 'BPPC_3_' in self.file:self.Max_iteration = 2000

        self.algorithm_type = algorithm_type
        self.plateau_threshold = plateau_threshold
        self.n_offsprings = n_offsprings
        self.n_partitions = n_partitions

        self.problem = None
        self.algorithm = None
        self.path = None
        # self.History = []
        self.res = None
        self.No_of_threads = No_of_threads
        self.dump_history = dump
        self.plt_convergence = plt_convergence
        self.save_initial_pop = save_initial_pop
        self.start_extra_improving = start_extra_improving
        self.save_history = save_history
        self.output_per_gen_file = None
        self.dic_ = {}

        self.seed = rand if rand is not None else rand

        print('algorithm_type = ', self.algorithm_type)
        if self.algorithm_type == 'Swarm':
            self.exp_id = '%s %s_%d_%s Pop_%d  %s' % (self.file, self.algorithm_type, self.updating_method, init_method_dic[self.initialization_method], self.pop_size, get_TimeStamp_str())
        elif self.algorithm_type == 'GA':
            self.exp_id = '%s %s_%s Pop_%d  %s' % ( self.file, self.algorithm_type, init_method_dic[self.initialization_method],
                                                    self.pop_size, get_TimeStamp_str())
        else:
            self.exp_id = '%s %s_%s Pop_%d  %s' % (self.file, self.algorithm_type, init_method_dic[self.initialization_method],
                                                    self.pop_size, get_TimeStamp_str())

    def run(self, N=1):
        try:
            if not os.path.isdir('exp_output'):
                os.mkdir('exp_output')
        except:
            ...

        os.mkdir('exp_output/'+self.exp_id)
        self.path = 'exp_output/%s/%s'%(self.exp_id, self.exp_id)
        self.seed = [np.random.randint(1000) for i in range(N)] if self.seed is None else self.seed
        print('self.seed = ', self.seed)
        write_to_file(self.path, "SEED:" + ', '.join([str(k) for k in self.seed]))

        cols = [ 'file', 'Optimal', '# Run_ID', 'random_seed', 'Fitness', 'Final_No_of_Bin', 'Time (min)', 'n_gen']
        csv_create_empty_df(self.path, cols)

        self.output_per_gen_file = self.path + '_OUTPUT'
        csv_create_empty_df(self.output_per_gen_file, ['Run_id', 'Iteration', 'Fitness', 'Bins' ])
        for i in range(N):
            a = time.time()
            res = self.run_single(self.seed[i], i)
            # print( '========================= res = ' , res )
            if res is None:
                write_to_file(self.path, '\n'*3+'===> res is none: Seed_%d'%self.seed[i]+'\n'*3)
                print('\n'*3+'===> res is none: Seed_%d'%self.seed[i]+'\n'*3)
                continue

            # self.History.append(self.res)

            res_record = [ self.file,
                           file_optimal[self.file],
                           i+1,
                           self.seed[i],
                           res.algorithm.opt.get('F')[0][0],
                           len(set(res.algorithm.opt.get('X').tolist()[0])),
                           np.round( (time.time() - a)/60, 3),
                           res.algorithm.n_gen,
                           # ','.join([ '%-.5f'%h.opt.get('F')[0][0] for h in res.history]),
                           # ','.join(['%d'% len(set(h.opt.get('X')[0])) for h in res.history]),
                   ]
            csv_append_row(self.path, res_record)

        try:
            if self.plt_convergence:
                plt_(self.output_per_gen_file)
        except Exception as exc:
            print('Error while Plotting')
            print('\n**** Err:\n', traceback.format_exc())

        if self.dump_history:
            self.dump()

    def run_single(self, rand, iteration):
        try:
            self.problem = BPP_wz_conlict(
                                            problem_file='BBP_Jully/data/%s.txt' % self.file,
                                            optimal_No_of_bins=self.optimal_No_of_bins,
                                            initialization_method=self.initialization_method,
                                         )


            if self.algorithm_type=='Swarm':
                save_initial_pop = '%s  -  run_%d seed_%d'%(self.path, iteration+1, rand) if self.save_initial_pop else None
                self.algorithm = Jellyfish(  pop_size=self.pop_size,
                                             updating_method=self.updating_method,
                                             update_Salah_No=self.update_Salah_No,
                                             # No_of_threads = self.No_of_threads,
                                             save_initial_pop = save_initial_pop,
                                             iteration = iteration+1,
                                             start_extra_improving = self.start_extra_improving,
                                             output_per_gen_file = self.output_per_gen_file
                                          )

            elif self.algorithm_type=='GA':
                ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=self.n_partitions)
                repair = MyRepair(self.problem, rand)
                self.algorithm = UNSGA3_modified(
                                            ref_dirs,
                                            iteration=iteration + 1,
                                            output_per_gen_file=self.output_per_gen_file,
                                            pop_size=self.pop_size,
                                            n_offsprings=self.n_offsprings,
                                            #### => selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                                            # sampling=get_sampling("int_random"),
                                            crossover=get_crossover("(bin|int)_hux"),  # "int_sbx"
                                            mutation=get_mutation("int_pm"),  # "bin_bitflip"
                                            repair=repair,
                                            eliminate_duplicates=True,
                                        )

            elif self.algorithm_type=='PSO':
                save_initial_pop = '%s  -  run_%d seed_%d' % (self.path, iteration + 1, rand) if self.save_initial_pop else None
                self.algorithm = PSO(  pop_size=self.pop_size,
                                       save_initial_pop=save_initial_pop,
                                       iteration=iteration + 1,
                                       output_per_gen_file=self.output_per_gen_file
                                           )

            elif self.algorithm_type=='Jaya':
                save_initial_pop = '%s  -  run_%d seed_%d' % (self.path, iteration + 1, rand) if self.save_initial_pop else None
                self.algorithm = Jaya( pop_size=self.pop_size,
                                       save_initial_pop=save_initial_pop,
                                       iteration=iteration + 1,
                                       output_per_gen_file=self.output_per_gen_file
                                           )

#####################################################################################################################################################
            self.algorithm.initialization = Fathalla_initialization()
            self.res =  minimize(
                                     self.problem,
                                     self.algorithm,
                                     verbose = 0,
                                     seed = rand,
                                     save_history = self.save_history,
                                     termination  = Fathalla_termination(self.Max_iteration, self.plateau_threshold),
                                )

            out_pop = '%s  -  run_%d seed_%d    -   Iteration_%d' % (self.path, iteration + 1, rand, self.res.algorithm.n_gen)
            dump(self.res.algorithm.pop.get('X'), out_pop)
            return self.res
        except Exception as exc:
            # s = '\n\n**** Err:  Gen_%d Seed_%d\n'%(  self.algorithm.n_gen, rand)  + traceback.format_exc()
            # write_to_file(self.path, s)
            # print(s)
            print(traceback.format_exc())
            sys.exit()


    def dump(self):
        path = self.path + '.pkl'
        try:
            if not os.path.isdir('exp_output/'):
                os.mkdir('exp_output/')
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            print('\n**** Err: A005 Location not found:\n', traceback.format_exc())
        except Exception as exc:
            print('\n**** Err:B005 Dumping Experiment Err:\n', traceback.format_exc())
        print('path = "%s"'%path)
        return path

    @staticmethod
    def load(path):
        return load_pkl(path)

    def get_seed(self):
        return self.seed # [h.algorithm.seed for h in self.History]

    def get_exp_properties(self):
        s = ''
        s += 'rand = %d'% self.seed[-2] + '\n'
        s += 'algorithm_type: ' + self.algorithm_type + '  Initialization: "%s"' % self.initialization_method  + '\n'
        s += 'File: "%s"' % self.file + '  No_Bins:%d' % self.problem.g.no_of_items+ \
              '  Bin_Capacity:%d' % self.problem.g.bin_capacity + '   OPTIMAL:', self.optimal_No_of_bins + '\n'
        s += 'Max_Degree:', self.problem.g.max_deg + '\n'
        s += 'OPTIMAL Fitness: %-.4f  No_oF_Bins: %d' % (
        self.res.algorithm.opt.get('F')[0][0], len(set(self.res.algorithm.opt.get('X').tolist()[0]))) + '\n'
        s += 'Run_time: %-.5f sec' % (self.res.exec_time / 1000) + '\n'
        print(s)
        return s