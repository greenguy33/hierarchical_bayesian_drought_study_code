This repository contains the data and code needed to reproduce the results of the paper "Disparities in the impact of drought on agriculture across countries" (Scientific Reports, '25)

Contact: Hayden Freedman (hfreedma@uci.edu)

To re-generate the figures and tables presented in the paper, run `scripts/gen_paper_results.py`. This file uses the pre-built Bayesian model files and regression dataset.

To re-generate the main dataset for the paper, run `scripts/create_regression_dataset.py`.

To run Bayesian model sampling, run `scripts/tfp_bayes_yfe_cre_for_drought_full.py`. Note that this is a long running script as Bayesian sampling must complete. Samples will be scaled to facilitate the sampler, and thus the output cannot be directly interpreted. See code block starting at line 96 of `scripts/gen_paper_results.py` for unscaling routine.

The results of this paper were also reproduced using the [_Climate Econometrics Toolkit_](https://github.com/greenguy33/climate_econometrics_toolkit), a tool currently under development to assist with Climate Econometrics study workflows. The reproduction can be found in this [notebook](https://github.com/greenguy33/climate_econometrics_toolkit/blob/main/notebooks/drought_paper_reproduction.ipynb).
