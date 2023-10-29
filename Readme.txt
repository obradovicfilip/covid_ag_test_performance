##### Bounding Sensitivity and Specificity of COVID-19 Antigen Tests #####
### Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu
Version: 1.0
Date: March 2022

This is an archive of Python scripts accompanying the paper. It contains all codes for result replication.
## Directory Structure

\covid_ag_test_performance
  replication_code.py - script for reproducing results in the paper
  calculator.py - script that allows to apply the method of the paper to other data
  functions.py - script containing functions
  package_installation.py - script for installing necessary packages

## Installation

The procedure requires packages numpy, scipy, statsmodels, matplotlib, and joblib.
If they are not already installed, you may install them by running the package_installation.py script.
Alternatively, you may enter the following commands in your terminal of choice if you use pip:

$ pip install joblib
$ pip install matplotlib
$ pip install scipy
$ pip install statsmodels
$ pip install numpy

## Replication

Simply run the replication script. It will print out all results in the paper.

## Using calculator.py

The procedure requires several computation-related parameters, as well as assumptions on the reference test.

# Computational Parameters

grid_steps - Number of points in the grid in EACH dimension for the parameters theta1 and theta0. Total number of points is
             grid_steps*grid_steps. Default value of 316 yields ~ 100000 points for theta values. Integer scalar.
gridsteps_s - Number of points in the grid in EACH dimension for S. Assumes S is rectangular. Integer scalar.
              Total number of points is gridsteps_s*gridsteps_s if both s1 and s0 are not known exactly. Otherwise, it is gridsteps_s.
boot_samples - Number of bootstrap draws for the Romano, Shaikh and Wolf (2014) testing procedure. Original paper uses 500.
               Integer scalar.
parallel - Parallelizes the procedure over multiple threads. May reduce computation time significantly. Boolean scalar.
num_threads -Sets number of used threads to maximum. Value -1 uses all available threads. Integer scalar.
graph_name = 'EUA_example' - Name of produced graph. Saved in the 'Graphs' folder. String.
include_apparent - Plots estimates and 95% Clopper-Pearson projection confidence sets for apparent measures.

# Test Parameters

wrongly_agree_0 - True if tests have a tendency to wrongly agree for y=0. Boolean scalar.
wrongly_agree_1 - True if tests have a tendency to wrongly agree for y=1. Boolean scalar.

s1 - Assumed sensitivity of the reference test. Can be a float in [0,1] or interval in the form of a list. (e.g. s1=0.9 or s1=[0.8,0.9])
s0 - Assumed specificity of the reference test. Can be a float in [0,1] or interval in the form of a list. (e.g. s0=0.9 or s0=[0.8,0.9])

# Study Data

t1r1 - Number of positives on both tests.
t1r0 - Number of participants positive on the index and negative on the reference test.
t0r1 - Number of participants negative on the index and positive on the reference test.
t0r0 - Number of negatives on both tests.

## Results

The function "calculate" produces a graph of the confidence set and the estimated identified set for sensitivity and specificity, 
and prints the projections of the bounds. Graphing is recommended as it perserves the structure of the identified set.

It outputs all points in the grid in the estimated identified set and the confidence set.

Procedure is computationally intensive. It can take up to several hours to complete depending on the available resources 
and chosen parameters. The given example takes 10 minutes to complete on a Ryzen 9 3900X with 12 cores.
    	

## Potential Issues

- Confidence set does not contain points in the estimated identified set. Probable reason - grid over parameter space too coarse.
  Try increasing grid_steps first, and gridsteps_s if the issue persists
- Error returned: "scipy.spatial.qhull.QhullError: QH6154 Qhull precision error: Initial simplex is flat (facet 1 is coplanar with the interior point)"
  Probable reason - grid over parameter space too coarse. Try increasing grid_steps.
- Lower bound is higher than the upper bound in the projected bounds. Check assumptions, some may not hold. This is not a sufficient conditon to
  refute the assumptions.
