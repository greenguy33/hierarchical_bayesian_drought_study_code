This repository contains the data and code needed to reproduce the results of the papers "The Global Effects of Drought on Agriculture: A Hierarchical Bayesian Approach" (in submission)

Contact: Hayden Freedman (hfreedma@uci.edu)

To re-generate the figures and tables presented in the paper, run `scripts/gen_paper_results.py`. This file uses the pre-built Bayesian model files and regression dataset.

To re-generate the main dataset for the paper, run `scripts/create_regression_datasets.py`.

To run Bayesian model sampling, run `scripts/tfp_bayes_yfe_cre_for_drought_full.py`. Note that this is a long running script as Bayesian sampling must complete.
