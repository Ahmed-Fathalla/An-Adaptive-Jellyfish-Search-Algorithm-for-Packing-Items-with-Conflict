# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief:
"""

from .base import *

class PSO(Algorithm, Base_):
    def __init__(self,
                 pop_size=25,
                 eliminate_duplicates=True,
                 save_initial_pop=None,
                 iteration='',
                 output_per_gen_file='',
                 **kwargs):

        super().__init__(**kwargs)
        self.F = None
        self.iteration = iteration
        self.pop_size = pop_size
        self.G_best = None
        self.L_best = None
        self.L_Worst = None
        self.item_max = None
        self.items = None
        self.bin_capacity = None
        self.initialization = None  # Fathalla_initialization()
        self.save_initial_pop = save_initial_pop
        self.output_per_gen_file = output_per_gen_file
        self.problem_file = None
        self.Optimal_solution = None
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()

        self.alfa_1 = self.alfa_2 = None

        self.V = []

    def _initialize_infill(self, infills=None, **kwargs):
        self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
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

        self.L_best = self.pop
        self.G_best = self.pop[  np.argmin(self.L_best.get('F'))  ]

        self.seed_ = self.seed
        self.Optimal_solution = self.problem.optimal_No_of_bins

        self.X = self.pop.get("X")
        self.initial_velocity()

        return self.pop

    def initial_velocity(self, number_of_swaps=3):
        self.V = (np.ones( shape=(self.X.shape[0],1), dtype=np.int )*-1).tolist()
        for i, x in enumerate(self.X):
            unique_bins = set(x)

            # Here: number_of_waps*2 to guarantee that there is not duplication
            swaps = np.random.choice(  range(len(x)),
                                       size=number_of_swaps,
                                       replace=False
                                    ).tolist()

            lst = []
            for itm in swaps:
                qq = np.random.choice( list(unique_bins-{x[itm]}))
                lst.append([ itm+1, x[itm], qq ])
            self.V[i] = lst

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
        print("\r", 'PSO ' + self.problem_file, '%s/%-4d  Best (bins:%d(%d), fitness:%-.5f)'%( str(self.iteration),self.n_gen,
                                                                                   len(set(self.G_best.get('X'))),
                                                                                   self.Optimal_solution,
                                                                                   self.G_best.get('F') ), end="")
        pop = self.pop
        self.X = pop.get("X")

        for i in range(len(self.X)):
            self.check_conf(self.X[i, :], 'Gen_%d indiv_%d/%d _step_2'%(self.n_gen, i, len(self.X)))

        self.off = np.ones(self.X.shape, dtype=int)*-100
        self.x_G_best = self.G_best.get('X')  # X[np.where(F == np.min(F))[0][0]]

        for i in range(len(self.X)):
            self.alfa_1 = rnd()
            self.alfa_2 = rnd()
            aa = self.dot_(self.alfa_1, self.minus_(self.L_best[i].get('X') , self.X[i])).tolist()
            bb = self.dot_(self.alfa_2, self.minus_(self.G_best.get('X') , self.X[i])).tolist()
            self.V[i] = [*self.V[i], *aa, *bb]
            self.off[i] = self.plus_( self.X[i], self.V[i] )

        self.off = Population.create(self.off.astype(int))
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        better_ = np.where(self.L_best.get('F') > self.off.get('F'))[0]
        self.L_best[better_] = self.off[better_]

        best_local_best = self.L_best[np.argmin(self.L_best.get('F'))]
        self.G_best = best_local_best if best_local_best.get('F') < self.G_best.get('F') else self.G_best

        self.pop = self.L_best

        gc.collect()
        return self.pop

    def minus_(self, a, b):
        index_of_non_similar_cols = list(np.where( (a - b)!=0 )[0] )
        if len(index_of_non_similar_cols)==0:return []
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
            if from_ == to_:continue
            if bins_[ to_ ].remaining >= items_dict[ item_ ].w and len(np.intersect1d( bins_[ to_ ].item_lst, items_dict[ item_ ].conflict_lst)) == 0:
                bins_[from_].free_an_item( item_ , items_dict[ item_ ].w )
                # add item to the new bin
                bins_[to_].pack_an_item( item_, items_dict[item_].w)
                items_dict[item_].packed_to_bin = to_
                sucess_swaps += 1

        # ===================
        # readjust_bins_order
        # ===================
        bins_, new_individual = self.readjust_bins_order(bins_)
        return new_individual

    def dot_(self, a, S):
        len_ = int(np.ceil(a * len(S)))
        indices = np.random.choice(list(range(0, len(S))), len_, replace=False)
        output = np.array(S)[ np.array(indices, dtype=int)]
        return output

    def _set_optimum(self, force=False):
        self.opt = filter_optimum(self.pop, least_infeasible=True)
        csv_append_row(self.output_per_gen_file, [self.iteration, self.n_gen, self.opt.get('F')[0][0], len(set(self.opt.get('X')[0]))])