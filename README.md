# GPforBO

Bayesian optimization (BO) is used to find a maximum or minimum global in a black box process through acquisition functions (AF). In this repo a genetic algorithm (GA) is implemented to find the best AF. The GA is tested with synthetic data in a problem with one dimension.

# Installation

You need anaconda to replicate this project. A yml file is available to replicate the environment. Just use the next command once you have anaconda or miniconda installed in your computer:

```
conda env create -n YOUR-ENVNAME --file environment.yml
```

# Report

A complete report using this algorthm with several experiments is available in this repo (proyml2_v1.pdf). An english version will be availeble soon, with instructions to test and generate another solutiions.

# Bugs report

In this current version of the code there are warnings for a zero division. 
Some minor bugs must be solved along with an error that stops the excecution of the code.