# An Adaptive Jellyfish Search Algorithm for Packing Items with Conflict  [(Go-to-Paper)](https://www.mdpi.com/2227-7390/11/14/3219]) [(Download-PDF)](https://www.mdpi.com/2227-7390/11/14/3219/pdf?version=1690003688)

## Abstract
The bin packing problem (BPP) is a classic combinatorial optimization problem with several variations. The BPP with conflicts (BPPCs) is not a well-investigated variation. In the BPPC, there are conditions that prevent packing some items together in the same bin. There are very limited efforts utilizing metaheuristic methods to address the BPPC. The current methods only pack the conflict items only and then start a new normal BPP for the non-conflict items; thus, there are two stages to address the BPPC. In this work, an adaption of the jellyfish metaheuristic has been proposed to solve the BPPC in one stage (i.e., packing the conflict and non-conflict items together) by defining the jellyfish operations in the context of the BPPC by proposing two solution representations. These representations frame the BPPC problem on two different levels: item-wise and bin-wise. In the item-wise solution representation, the adapted jellyfish metaheuristic updates the solutions through a set of item swaps without any preference for the bins. In the bin-wise solution representation, the metaheuristic method selects a set of bins, and then it performs the item swaps from these selected bins only. The proposed method was thoroughly benchmarked on a standard dataset and compared against the well-known PSO, Jaya, and heuristics. The obtained results revealed that the proposed methods outperformed the other comparison methods in terms of the number of bins and the average bin utilization. In addition, the proposed method achieved the lowest deviation rate from the lowest bound of the standard dataset relative to the other methods of comparison. 

## Running an Experiment
```python
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
```

## Citing

```bibtex
@Article{math11143219,
AUTHOR = {El-Ashmawi, Walaa H. and Salah, Ahmad and Bekhit, Mahmoud and Xiao, Guoqing and Al Ruqeishi, Khalil and Fathalla, Ahmed},
TITLE = {An Adaptive Jellyfish Search Algorithm for Packing Items with Conflict},
JOURNAL = {Mathematics},
VOLUME = {11},
YEAR = {2023},
NUMBER = {14},
ARTICLE-NUMBER = {3219},
URL = {https://www.mdpi.com/2227-7390/11/14/3219},
ISSN = {2227-7390},
DOI = {10.3390/math11143219}
}
```
[paper]: https://www.mdpi.com/2227-7390/11/14/3219
