# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

from .base import *

class Jellyfish_base(Algorithm, Base_):
    def __init__(self,
                 pop_size=25,
                 trend_type = 'random',
                 eliminate_duplicates=True,
                 updating_method = 1,
                 update_Salah_No = 1,
                 extra_improving_ratio = 0.2,
                 # No_of_threads=5,
                 save_initial_pop = None,
                 iteration = '',
                 start_extra_improving = 300,
                 output_per_gen_file = '',
                 **kwargs):

        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way

        super().__init__(**kwargs)

        self.opt__ = None
        self.F = None
        self.iteration = iteration

        self.u, self.l = 0, 0
        self.ct = 0.0
        self.x_mean = []
        self.trend_type = trend_type
        self.pop_size = pop_size
        self.G_best = None
        self.L_best = None
        self.item_max = None
        self.items = None
        self.bin_capacity = None
        self.swaps_lst = []
        self.initialization = None # Fathalla_initialization()
        self.save_initial_pop = save_initial_pop

        self.update_Salah_No = update_Salah_No
        self.updating_method = updating_method
        self.extra_improving_ratio = extra_improving_ratio

        self.start_extra_improving = start_extra_improving
        self.output_per_gen_file = output_per_gen_file

        self.problem_file = None
        self.Optimal_solution = None
        # self.No_of_threads = No_of_threads
        # self.number_of_individual_per_thread = int(self.pop_size / self.No_of_threads)
        # print('Threading: No:%d  number_of_individual_per_thread:%d' % (self.No_of_threads, self.number_of_individual_per_thread))

        self.enable_trix = False

        self.task_1 = [self.item_wise_1, self.bin_wise_1][self.updating_method-1]
        self.task_2 = [self.item_wise_2, self.bin_wise_2][self.updating_method-1]
        self.task_3 = [self.item_wise_3, self.bin_wise_3][self.updating_method-1]

        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()

    def _initialize_infill(self, infills=None, **kwargs):
        self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        # print('self.termination = ', self.termination)
        # print('self.initialization = ', self.initialization)
        self.problem_file = self.problem.g.problem_file.split('/')[-1][:-4]

        self.pop = self.pop[:self.pop_size]
        self.evaluator.eval(self.problem, self.pop, algorithm=self)

        if self.save_initial_pop is not None:
            dump(self.pop.get('X'), self.save_initial_pop)

        self.bin_capacity = self.problem.g.bin_capacity
        self.items = self.problem.g.items
        self.item_dict_empty_bins = self.problem.g.item_dict_empty_bins

        self.empty_bins = copy.deepcopy(self.problem.g.empty_bins)
        self.item_max = np.max( list(self.items.keys()) )

        self.G_best = self.L_best = self.pop[np.argmin(self.pop.get('F'))]

        # self.run_id = self.iteration

        self.seed_ = self.seed
        self.Optimal_solution = self.problem.optimal_No_of_bins

        # self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return self.pop
        # return self.initialization.do(self.problem, self.pop_size, algorithm=self)

    def next(self):
        infills = self.infill()  # get the infill solutions

        # call the advance with them after evaluation
        if infills is not None:
            # self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()

    def _infill(self):
        if self.n_gen > self.start_extra_improving:
            self.enable_trix = True

        # print('===============', self.G_best.get('X','F'))
        print("\r", 'Swarm ' + self.problem_file, '%s/%-4d  Best (bins:%d(%d), fitness:%-.5f)'%( str(self.iteration),self.n_gen,
                                                                                   len(set(self.G_best.get('X'))),
                                                                                   self.Optimal_solution,
                                                                                   self.G_best.get('F') ), end="")

        pop = self.pop

        self.X, self.F = pop.get("X", "F")
        # print('\n'*5,'self.X = ', self.X, '='*70)

        self.L_best = pop[np.argmin(pop.get('F'))]
        if self.L_best.get('F') < self.G_best.get('F'):
            self.G_best = copy.deepcopy(self.L_best)

        for i in range(len(self.X)):
            # print('i====>', i)
            # print('self.X[i, :] = ', self.X[i, :])
            self.check_conf(self.X[i, :], 'Gen_%d indiv_%d/%d _step_2'%(self.n_gen, i, len(self.X)))
            # print('i====>', i)

        self.off = np.ones(self.X.shape, dtype=int)*-100
        self.x_G_best = self.G_best.get('X')  # X[np.where(F == np.min(F))[0][0]]
        self.distance_to_mean = np.abs(self.F - np.mean(self.F))

        for i in range(len(self.X)):
            # print('=========== i = ', i)
            self.ct = np.abs((1 - self.n_gen / self.termination.n_max_gen) * (2 * rnd() - 1))
            if self.ct < 0.5:
                # print('Gen_%d'%self.n_gen, i, 'A')
                if self.trend_type == 'random':
                    self.x_mean = self.problem.g.initialize_pop(1)[0]
                elif self.trend_type == 'mean':
                    self.x_mean = self.X[np.argmin(self.distance_to_mean)]
                elif self.trend_type == 'median':
                    self.x_median = self.X[np.where(self.F == np.median(self.F))[0][0]]
                # print('self.task_1 = ')

                self.off[i, :] = self.task_1(self.X[i, :])

                # print('self.off[i, :] = ', self.off[i, :])
                self.check_conf(self.off[i, :], 'C')
            else:
                r = rnd()
                if r > self.ct:
                    # print('Gen_%d' % self.n_gen, i, 'B')
                    # print('self.task_2 = ')
                    self.off[i, :] = self.task_2(self.X[i, :])
                    # print('Gen_%d' % self.n_gen, i, 'B1')
                    self.check_conf(self.off[i, :], 'EE')
                    # print('Gen_%d' % self.n_gen, i, 'B2')
                else:
                    # print('Gen_%d' % self.n_gen, i, 'C')
                    j = np.random.randint(low=0, high=len(self.F))
                    # print('self.task_3 = ')
                    self.off[i, :] = self.task_3(self.X[i, :], self.X[j, :], i, j)
                    self.check_conf(self.off[i, :], 'FF')

        # print('self.off = ', self.off)
        self.off = Population.create(self.off.astype(int))
        # print(' ----------', type(self.off))
        # print('self.off = ', self.off)

        # return self.off
        self.evaluator.eval(self.problem, self.off, algorithm=self)
        # print('3 pop = ', self.off.get('F'))

        better_ = np.where(self.pop.get('F') > self.off.get('F'))[0]

        # print('self.off = ', self.off.get('X'))
        # print('better_ = ', better_)

        self.pop[better_] = self.off[better_]
        # print('\n'*5, self.pop.get('X'),'==='*10,'\n'*2)

        gc.collect()
        return self.pop


    def item_wise_1(self, X):
        trend = self.minus_(self.x_G_best, self.x_mean)
        # $ print('\ttrend = ', trend)
        return self.plus_(X, self.dot_(rnd(), trend))  # .astype(int)

    def bin_wise_1(self, X):
        trend = self.interact_2(self.x_G_best, self.x_mean)
        return self.interact_2(X, trend) # self.interact_2(X, trend) # self.interact_2(X, trend)

    def item_wise_2(self, X):
        s_len = np.random.randint(low=1, high=self.item_max + 1)
        p = np.random.randint(low=1, high=self.item_max + 1, size=(s_len, 1))

        bin_max = len(set(X))  # .astype(int)
        q_r = np.random.randint(low=1, high=bin_max + 1, size=(s_len, 2))
        s = np.hstack((p, q_r))

        return self.plus_(X, self.dot_(rnd(), s))  # .astype(int)

    def bin_wise_2(self, X):
        return self.interact_1(X)

    def item_wise_3(self, X_i, X_j, i, j):
        direction = self.minus_(X_i, X_j) if self.F[j] > self.F[i] else self.minus_(X_j,X_i)  # .astype(int) # .astype(int)
        return self.plus_(X_i, self.dot_(rnd(), direction))  # .astype(int)

    def bin_wise_3(self, X_i, X_j, i, j):
        direction = self.interact_2(X_i, X_j)
        return self.interact_2(X_i, direction)

    def _set_optimum(self, force=False):
        # print('self.pop = ', self.pop, self.n_gen)
        self.opt = filter_optimum(self.pop, least_infeasible=True)
        # print('self.opt.get("F")[0] = ', self.opt.get("F")[0])
        # print('self.opt.get("X")[0] = ', self.opt.get("X")[0])
        # print('self.opt = ', self.opt)

        csv_append_row(self.output_per_gen_file, [self.iteration, self.n_gen, self.opt.get('F')[0][0], len(set(self.opt.get('X')[0]))])

    #####################################################################################33
    def interact_1(self, x):
        ...

    def interact_2(self,a, b):
        ...

    def minus_(self, a, b):
        ...

    def plus_(self, x, S):
        ...

    def dot_(self, a, S):
        ...
