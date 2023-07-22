from BBP_Jully import *

e = Exp( 
            file = 'BPPC_1_1_1',
            initialization_method = 'random_random',
            pop_size = 25,
            Max_iteration = 20,
            algorithm_type = 'Swarm',
            updating_method = 1,
            update_Salah_No=1 ,
            plateau_threshold=15,
            # rand = 48,
        )
e.run(5)