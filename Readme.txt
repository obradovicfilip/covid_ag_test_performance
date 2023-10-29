##### Bounding Sensitivity and Specificity of COVID-19 Antigen Tests #####
### Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu
Version: 1.1
Date: October 2023

This is an archive of Python scripts accompanying the paper. It contains all codes for result replication.
## Directory Structure

\covid_ag_test_performance
  replication_code.py - script for reproducing results in the paper
  functions.py - script containing functions
  package_installation.py - script for installing necessary packages

NOTE: A script that allows easy application of the method can be found following - https://github.com/obradovicfilip/bounding_test_performance

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

The procedure is computationally intensive. It can take up to several hours to complete depending on the available resources 
and chosen parameters. The given example takes 10 minutes to complete on a Ryzen 9 3900X with 12 cores.

# Computational Parameters

grid_steps - Number of points in the grid in EACH dimension for the parameters theta1 and theta0. Total number of points is
             grid_steps*grid_steps. Default value of 316 yields ~ 100000 points for theta values. Integer scalar.

s1 - Assumed sensitivity of the reference test. Can be a float in [0,1] or interval in the form of a list. (e.g. s1=0.9 or s1=[0.8,0.9])
s0 - Assumed specificity of the reference test. Can be a float in [0,1] or interval in the form of a list. (e.g. s0=0.9 or s0=[0.8,0.9])

# Study Data

t1r1 - Number of positives on both tests.
t1r0 - Number of participants positive on the index and negative on the reference test.
t0r1 - Number of participants negative on the index and positive on the reference test.
t0r0 - Number of negatives on both tests.

    	
