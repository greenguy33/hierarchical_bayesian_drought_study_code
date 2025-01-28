import pandas as pd
import numpy as np
import pickle as pkl
import pymc as pm
from pytensor import tensor as pt 
from sklearn.preprocessing import StandardScaler
import copy
import arviz as az


tfp_regression_data_insample = pd.read_csv("data/regression/tfp_regression_data.csv").dropna().reset_index(drop=True)
countries_in_dataset = set(tfp_regression_data_insample.country)
years_in_dataset = set(tfp_regression_data_insample.year)
country_fe_cols = [col for col in tfp_regression_data_insample.columns if "country_fixed_effect" in col]
year_fe_cols = [col for col in tfp_regression_data_insample.columns if "year_fixed_effect" in col]

countries_to_remove = []
for country_fe in country_fe_cols:
    country = country_fe.split("_")[0]
    if country not in countries_in_dataset:
        tfp_regression_data_insample = tfp_regression_data_insample.drop(country_fe, axis=1)
        countries_to_remove.append(country)
for country in countries_to_remove:
    country_fe_cols.remove(country+"_country_fixed_effect")

for year_fe in year_fe_cols:
    year = year_fe.split("_")[0]
    if int(year) not in years_in_dataset:
        tfp_regression_data_insample = tfp_regression_data_insample.drop(year_fe, axis=1)

model_spec = {
    "continuous_covariates" : [
        'temp_[weight]',
        'temp_[weight]_2'
    ],
    "discrete_covariates" : ['drought'],
    "fixed_effects" : ["year"],
    "weights" : "ag_weighted",
    "target" : "fd_ln_tfp"
}

covar_scalers = []
for covar_col in model_spec["continuous_covariates"]:
    covar_scalers.append(StandardScaler())
    tfp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])+"_scaled"] = covar_scalers[-1].fit_transform(np.array(tfp_regression_data_insample[covar_col.replace("[weight]",model_spec["weights"])]).reshape(-1,1)).flatten()
target_var_scaler = StandardScaler()
tfp_regression_data_insample[model_spec["target"]+"_scaled"] = target_var_scaler.fit_transform(np.array(tfp_regression_data_insample[model_spec["target"]]).reshape(-1,1)).flatten()

target_data = tfp_regression_data_insample[model_spec["target"]+"_scaled"]
model_variables = []
for covar in model_spec["continuous_covariates"]:
    model_variables.append(covar.replace("[weight]",model_spec["weights"])+"_scaled")
for covar in model_spec["discrete_covariates"]:
    model_variables.append(covar)
for fe in model_spec["fixed_effects"]:
    for fe_col in [col for col in tfp_regression_data_insample.columns if col.endswith(f"{fe}_fixed_effect")]:
        model_variables.append(fe_col)
model_data = tfp_regression_data_insample[model_variables]

first_year_fe_col = [col for col in model_data.columns if "year_fixed_effect" in col][0]

model_data_first_fe_removed = copy.deepcopy(model_data)
model_data_first_fe_removed[first_year_fe_col] = 0

year_fe_vars = [col for col in model_data_first_fe_removed.columns if "year_fixed_effect" in col]
country_fe_vars = [col for col in model_data_first_fe_removed.columns if "country_fixed_effect" in col]

vars_to_bundle = []
for var in model_spec["continuous_covariates"]:
    vars_to_bundle.append(var.replace("[weight]",model_spec["weights"])+"_scaled")
if "country" in model_spec["fixed_effects"]:
    vars_to_bundle.extend(country_fe_vars)
if "year" in model_spec["fixed_effects"]:
    vars_to_bundle.extend(year_fe_vars)

print(f"Including variables: {vars_to_bundle}")

with pm.Model() as pymc_model:
    
    global_country_rs_mean = pm.Normal("global_country_rs_mean",0,10)
    global_country_rs_sd = pm.HalfNormal("global_country_rs_sd",10)
    country_rs_means = pm.Normal("country_rs_means", global_country_rs_mean, global_country_rs_sd, shape=(1,len(country_fe_cols)))
    country_rs_sd = pm.HalfNormal("country_rs_sd", 10)
    country_rs_coefs = pm.Normal("country_rs_coefs", country_rs_means, country_rs_sd)
    country_rs_matrix = pm.Deterministic("country_rs_matrix", pt.sum(country_rs_coefs * tfp_regression_data_insample[country_fe_cols],axis=1))

    drought_terms = pm.Deterministic("drought_terms", country_rs_matrix * model_data_first_fe_removed["drought"])

    covar_coefficients = pm.Normal("model_variable_coefs", 0, 5, shape=len(vars_to_bundle))
    covar_terms = pm.Deterministic("model_variable_terms", pt.sum(covar_coefficients * model_data_first_fe_removed[vars_to_bundle], axis=1))
    
    tfp_prior = pm.Deterministic(
        "tfp_prior",
        covar_terms +
        drought_terms
    )
    
    tfp_std_scale = pm.HalfNormal("tfp_std_scale", 10)
    tfp_std = pm.HalfNormal("tfp_std", sigma=tfp_std_scale)
    tfp_posterior = pm.Normal('tfp_posterior', mu=tfp_prior, sigma=tfp_std, observed=target_data)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4, tune=5000, draws=5000)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with open ('output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full_10k.pkl', 'wb') as buff:
    pkl.dump({
        "prior":prior,
        "trace":trace,
        "posterior":posterior,
        "var_list":list(model_data.columns),
        "model_spec":model_spec
    },buff)

print(az.summary(trace, var_names=["global_country_rs_sd", "country_rs_sd", "country_rs_coefs"]))
