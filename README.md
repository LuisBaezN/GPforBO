# GPforBO

Bayesian optimization (BO) is used to find a maximum or minimum global in a black box process through acquisition functions (AF). In this repo a genetic algorithm (GA) is implemented to find the best AF. The GA is tested with synthetic data in a problem with one dimension.

## Attention

The first version of the genetic algorithm is working. The bayesian optimization algorithm needs revision to its posterior conjunction to the GA.

# Installation

You need anaconda to replicate this project. A yml file is available to replicate the environment. Just use the next command once you have anaconda or miniconda installed in your computer:

```
conda env create -n YOUR-ENVNAME --file environment.yml
```