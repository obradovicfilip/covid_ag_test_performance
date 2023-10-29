##### Measuring Diagnostic Test Performance Using Imperfect Reference Tests: A Partial Identification Approach #####
### Replication script
## Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu

""" Replication script for generating all results in the paper.

    Procedure is computationally intensive. The script can take several hours to complete depending on the available
    resources.
"""


############################################ Import Functions ##########################################################

import functions

############################################# Set Parameters ###########################################################

seed = 100           # Set seed for replicability
s1 = 0.9             # Reference test sensitivity
s0 = 1               # Reference test specificity
grid_steps = 316     # 316*316 grid imposed over parameter space for theta
gridsteps_s = 10     # 10 points imposed over set S, if not a singleton
                     # In total 316*316*10 points in the grid over parameter space.

############################################### Study Data #############################################################

## Data:

# BinaxNow - https://www.fda.gov/media/141570/download?attachment
# QuickVue - https://www.fda.gov/media/144668/download?attachment
# iHealth - https://www.fda.gov/media/153923/download?attachment
# InteliSwab - https://www.fda.gov/media/149906/download?attachment


data = [['BinaxNow', 99, 5, 18, 338], ['iHealth', 33, 2, 2, 102],
        ['QuickVue', 56, 1, 2, 135],['InteliSwab', 52, 2, 9, 102]]
# Data in the form ['test', positives on both tests - t1r1, positives only on index - t1r0,
#               positives only on reference - t0r1, negatives on both tests - t0r0]

############################################### Computation ############################################################


for d in data:
    print("Test ", d[0])
    print("Sensitivity and specificity of reference ", s1, s0)
    graph_name,t1r1,t1r0,t0r1,t0r0 = d
    graph_name = graph_name+"_known_exact"
    functions.calculate(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=False,
                        grid_steps_conf=grid_steps, gridsteps_s=gridsteps_s,
                        method='2', boot_samples=500, parallel=True, num_threads=-1, filename=graph_name)




