import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import geopandas
import seaborn as sns
import mapclassify

data = pd.read_csv("data/regression/tfp_regression_data.csv").dropna().reset_index(drop=True)

model0_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_full_with_intercept_no_weather_vars_multiyear_drought_only_country_coefs.pkl'
model1_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_full_with_intercept_temp_multiyear_drought_only_country_coefs.pkl'
model2_file = 'output/models/bayes_models/tfp_bayes_yfe_cre_for_drought_full/tfp_bayes_yfe_cre_for_drought_full_with_intercept_temp_and_precip_multiyear_drought_only_country_coefs.pkl'

model0 = pd.read_pickle(model0_file)
model1 = pd.read_pickle(model1_file)
model2 = pd.read_pickle(model2_file)

plotFig1 = False
plotFig2 = False
plotFig3 = False
plotFig4 = False
plotFig5 = False
plotSupFig1 = False
plotSupFig2 = False
plotSupFig3 = True

genSupTab1 = False
genSupTab2 = False
genSupTab3 = False

# add total droughts by country and region to dataset
land_area_data = pd.read_csv("data/land-area-km.csv")
country_land_area = {}
for row in land_area_data.iterrows():
    row = row[1]
    country_land_area[row["Code"]] = row["Land area (sq. km)"]
droughts_by_country = {}
droughts_by_country_land_weighted = {}
for country in set(data.country):
    droughts_by_country[country] = np.count_nonzero(data.loc[(data.country == country)].drought)
    if country in country_land_area:
        droughts_by_country_land_weighted[country] = np.count_nonzero(data.loc[(data.country == country)].drought) / country_land_area[country]
    else:
        droughts_by_country_land_weighted[country] = np.NaN
data["total_drought_by_country"] = list(map(lambda x : droughts_by_country[x], data.country))

droughts_by_region = {}
for region in set(data.region23):
    droughts_by_region[region] = np.count_nonzero(data.loc[(data.region23 == region)].drought)
data["total_drought_by_region"] = list(map(lambda x : droughts_by_region[x], data.region23))

# find countries that have no drought in dataset
countries_with_no_drought = []
for country in set(data.country):
    if all(data[data.country == country].drought == 0):
        countries_with_no_drought.append(country)

# get global and regional ag. revenue share weights
revenue_data = pd.read_csv("data/revenue_shares.csv")
country_weights = {}
for row in revenue_data.itertuples():
    if row[3] in set(data.country):
        country_weights[row[3]] = np.mean([row[5],row[6],row[7],row[8],row[9],row[10]])
weight_sum = sum(list(country_weights.values()))
for country, val in country_weights.items():
    country_weights[country] = val/weight_sum
data["global_ag_weights"] = list(map(lambda x : country_weights[x], data.country))

regional_country_weights = {}
for region in set(data.region23):
    countries = set(data.loc[data.region23 == region].country)
    sum_of_group_weights = 0
    for country in countries:
        sum_of_group_weights += country_weights[country]
    for country in countries:
        regional_country_weights[country] = country_weights[country]/sum_of_group_weights
data["regional_ag_weights"] = list(map(lambda x : regional_country_weights[x], data.country))

# import developed vs. developing country data
country_development_classification = pd.read_csv("data/developed_developing_countries_UN.csv")
country_development_classification = {row[1]["ISO-alpha3 Code"]:row[1]["Developed / Developing regions"] for row in country_development_classification.iterrows()}
# removing Taiwan because it is not in country development data
data_mod = data[~data.country.isin(["TWN"])]
development_classification = []
for row in data_mod.iterrows():
    row = row[1]
    development_classification.append(country_development_classification[row.country])
data_mod["development"] = development_classification

model_impacts = {}
for model_index, model in enumerate([model0, model1, model2]):

    # unscale country coefficients
    scaled_vars = {}
    unscaled_vars = {}
    for country_index, var in enumerate(model["var_list"][-163:]):
        scaled_vars[var] = model["posterior"][:,:,:,country_index].data.flatten()
    for var, samples in scaled_vars.items():
        unscaled_vars[var] = np.array(samples) * np.std(data.fd_ln_tfp)

    # compute probability that drought has decreased TFP for each country
    country_percentiles = {}
    for country in list(model["var_list"][-163:]):
        country_percentiles[country.split("_")[0]] = len([sample for sample in unscaled_vars[country] if sample < 0])/len(unscaled_vars[country])

    # compute % impacts by country
    effect_by_country = {}
    for country in set(data.country):
        effect_by_country[country] = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for i in range(droughts_by_country[country]):
            effect_by_country[country] += unscaled_vars[country+"_country_fixed_effect"]
    percent_loss_by_country = {
        country:np.array([math.expm1(val)*100 for val in effect_by_country[country]])
        for country in set(data.country)
    }

    # compute % impacts by region
    percent_loss_by_region = {}
    for region in set(data.region23):
        countries = set(data.loc[data.region23 == region].country)
        sum_of_group_weights = 0
        for country in countries:
            sum_of_group_weights += country_weights[country]
        group_country_weights = {country:country_weights[country]/sum_of_group_weights for country in countries}
        region_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for country in countries:
            country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
            for i in range(droughts_by_country[country]):
                country_effect += unscaled_vars[country+"_country_fixed_effect"] * group_country_weights[country]
            if not all(country_effect) == 0:
                region_effect += country_effect
        percent_loss_by_region[region] = [math.expm1(val)*100 for val in region_effect]

    # compute probability that drought has decreased TFP for each region
    region_percentiles = {}
    for region in set(data.region23):
        region_percentiles[region] = len([val for val in percent_loss_by_region[region] if val < 0])/4000

    # compute global % impact
    global_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
    for country in set(data.country):
        country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for i in range(droughts_by_country[country]):
            country_effect += unscaled_vars[country+"_country_fixed_effect"] * country_weights[country]
        for i, val in enumerate(country_effect):
            global_effect[i] += val
    global_effect_percent = [math.expm1(val)*100 for val in global_effect]

    # compute % impacts for developed vs. developing countries
    development_effect_percent = {}
    for development in ["Developing","Developed"]:
        countries = [country for country in set(data_mod.country) if country_development_classification[country] == development]
        sum_of_group_weights = 0
        for country in countries:
            sum_of_group_weights += country_weights[country]
        group_country_weights = {country:country_weights[country]/sum_of_group_weights for country in countries}
        development_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
        for country in countries:
            country_effect = [0]*len(unscaled_vars["AFG_country_fixed_effect"])
            for i in range(droughts_by_country[country]):
                country_effect += unscaled_vars[country+"_country_fixed_effect"] * group_country_weights[country]
            for i, val in enumerate(country_effect):
                development_effect[i] += val
        development_effect_percent[development] = [math.expm1(val)*100 for val in development_effect]

    # add model results to dict
    model_data = {
        "country_percentiles":country_percentiles,
        "percent_loss_by_country":percent_loss_by_country,
        "region_percentiles":region_percentiles,
        "percent_loss_by_region":percent_loss_by_region,
        "global_percent_loss":global_effect_percent,
        "development_percent_loss":development_effect_percent,
        "country_coefficients":unscaled_vars
    }
    model_impacts[f"model{model_index}"] = model_data

# choose model to use to generate figures
model = "model1"
model_data = model_impacts[model]
data["country_drought_bin"] = list(map(lambda x : model_data["country_percentiles"][x], data.country))
data["region_drought_bin"] = list(map(lambda x : model_data["region_percentiles"][x], data.region23))

# plot Figure 1`
if plotFig1:
    fig, axes = plt.subplots(1,2,figsize=(15,15))
    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    data_ndcr = data[~data.country.isin(countries_with_no_drought)]
    country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axes[0])
    country_geopandas = country_geopandas.merge(
        data_ndcr,
        how='inner', 
        left_on=['iso_a3'],
        right_on=['country']
    )
    res = country_geopandas.plot(column="total_drought_by_country", cmap="RdYlGn_r", ax=axes[0], legend=True, legend_kwds={"location":"bottom", "pad":.04})
    axes[0].set_title("Total Droughts by Country (1961 - 2021)", size=20, weight="bold")
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_xlabel("Number of Droughts", size=20, weight="bold")
    axes[0].figure.axes[1].tick_params(labelsize=50)
    res.figure.axes[-1].tick_params(labelsize=15)

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    data_ndcr = data[~data.country.isin(countries_with_no_drought)]
    country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axes[1])
    country_geopandas = country_geopandas.merge(
        data_ndcr,
        how='inner',
        left_on=['iso_a3'],
        right_on=['country']
    )
    res = country_geopandas.plot(column="country_drought_bin", cmap="RdYlGn_r", ax=axes[1], legend=True, legend_kwds={"location":"bottom", "pad":.04})

    cmap = plt.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(vmin=country_geopandas['country_drought_bin'].min(), vmax=country_geopandas['country_drought_bin'].max())
    colors = [cmap(norm(value)) for value in country_geopandas['country_drought_bin']]
    axes[1].set_title("Country Prob. that drought has \n decreased TFP (Map)", size=20, weight="bold")
    axes[1].set_xlabel("Percentage (%)", size=20, weight="bold")
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    res.figure.axes[-1].tick_params(labelsize=15)

    plt.savefig("figures/drought_fig1.png", bbox_inches='tight')

# plot figure 2
if plotFig2:

    fig, axis = plt.subplots(3,1, figsize=(12,12))

    axes = [axis[0], axis[1], axis[2]]
    thresholds = [None, .84, .16]
    threshold_labels = ["Maximum-Likelihood", "Upper-Bound", "Lower-Bound"]

    for threshold, label, axis in zip(thresholds, threshold_labels, axes):
        country_percent_loss = {}
        for country, samples in model_data["percent_loss_by_country"].items():
            sorted_samples = sorted(samples)
            if threshold != None:
                country_percent_loss[country] = np.quantile(sorted_samples, threshold)
            else:
                country_percent_loss[country] = np.mean(sorted_samples)
        
        data[f"country_percent_loss_{label}"] = list(map(lambda x : country_percent_loss[x], data.country))
        
        country_geopandas = geopandas.read_file(
            geopandas.datasets.get_path('naturalearth_lowres')
        )
        data_ndcr = data[~data.country.isin(countries_with_no_drought)]
        country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axis)
        country_geopandas = country_geopandas.merge(
            data_ndcr,
            how='inner',
            left_on=['iso_a3'],
            right_on=['country']
        )
        bins = list(mapclassify.Quantiles(country_geopandas[f"country_percent_loss_Maximum-Likelihood"], k=10).bins)
        # delete bins that have similar range for map readability
        bins_to_remove = []
        for index, bin in enumerate(bins):
            if index != 0:
                if round(bin) == round(bins[index-1]):
                    bins_to_remove.append(index)
        bins = [val for index, val in enumerate(bins) if index not in bins_to_remove]
        legend_labels = []
        bins_as_int = [-1*round(bin) for bin in bins]
        for index, bin in enumerate(bins_as_int):
            if index == 0:
                legend_labels.append(f" > {bin} %")
            elif index == len(bins_as_int) - 1:
                legend_labels.append(f" < {bins_as_int[index-1]} %")
                break
            else:
                legend_labels.append(f"{bins_as_int[index-1]} % - {bin} %")
        
        country_geopandas.plot(column=f"country_percent_loss_{label}", cmap='RdYlGn', scheme="User_Defined", classification_kwds=dict(bins=bins), ax=axis)
        
        cmap = cm.get_cmap('RdYlGn')
        legend_bins = []
        for index, legend_label in enumerate(legend_labels):
            legend_bins.append(mpatches.Patch(color=cmap((1/len(legend_labels)*index)), label=legend_label))
        axis.legend(handles=legend_bins)
        
        axis.set_title(f"{label} Historical TFP Loss from Drought by country", size=15, weight="bold")
        axis.set_yticklabels([])
        axis.set_xticklabels([])

    plt.tight_layout()
    plt.savefig("figures/drought_fig2.png", bbox_inches='tight')

# plot figure 3
if plotFig3:

    fig = plt.figure(constrained_layout=True, figsize=(15,15))
    axs = fig.subplot_mosaic([['Top', 'Top'],['BottomLeft', 'BottomRight']], gridspec_kw={'width_ratios':[1,1], 'height_ratios':[1,.5]})

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    # data_ndcr = data[~data.country.isin(countries_with_no_drought)]
    # country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axs["BottomLeft"])
    country_geopandas = country_geopandas.merge(
        data,
        how='inner', 
        left_on=['iso_a3'],
        right_on=['country']
    )
    res = country_geopandas.plot(column="total_drought_by_region", cmap="RdYlGn_r", ax=axs["BottomLeft"], legend=True, legend_kwds={"location":"bottom"})
    axs["BottomLeft"].set_title("Total Droughts by Region (1961 - 2021)", size=20, weight="bold")
    axs["BottomLeft"].set_xticklabels([])
    axs["BottomLeft"].set_yticklabels([])
    axs["BottomLeft"].set_xlabel("Number of Droughts", size=20, weight="bold")
    axs["BottomLeft"].figure.axes[1].tick_params(labelsize=50)
    res.figure.axes[-1].tick_params(labelsize=20)

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    # data_ndcr = data[~data.country.isin(countries_with_no_drought)]
    # country_geopandas[country_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axs["BottomRight"])
    country_geopandas = country_geopandas.merge(
        data,
        how='inner',
        left_on=['iso_a3'],
        right_on=['country']
    )
    res = country_geopandas.plot(column="region_drought_bin", cmap="RdYlGn_r", ax=axs["BottomRight"], legend=True, legend_kwds={"location":"bottom"})
    cmap = plt.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(vmin=country_geopandas['region_drought_bin'].min(), vmax=country_geopandas['region_drought_bin'].max())
    colors = [cmap(norm(value)) for value in country_geopandas['region_drought_bin']]
    axs["BottomRight"].set_title("Regional Prob. that drought has \n decreased TFP (Map)", size=20, weight="bold")
    axs["BottomRight"].set_xlabel("Percentage (%)", size=20, weight="bold")
    axs["BottomRight"].set_xticklabels([])
    axs["BottomRight"].set_yticklabels([])
    res.figure.axes[-1].tick_params(labelsize=20)

    sorted_region_percentiles = dict(sorted(model_data["region_percentiles"].items(), key=lambda x: x[1], reverse=True))
    region_colors = []
    for region in sorted_region_percentiles.keys():
        region_index = country_geopandas.loc[country_geopandas.region23 == region].index[0]
        region_colors.append(colors[region_index])

    barcounts = [val*100 for val in list(sorted_region_percentiles.values())]
    barplot = axs["Top"].barh(list(range(0,20)), barcounts, color=region_colors)
    barcolors = ["white","white","white","white","white","black","black","black","black","black","black","black","black","black","black","black","black","black","black","white"]
    barlabels = axs["Top"].bar_label(barplot, list(sorted_region_percentiles.keys()), label_type = "center")
    barpaddings = list(reversed([60,-7,3,-79,-121,-105,-107,-103,-113,-142,-144,-155,-159,-170,-155,-168,-181,-195,-208,-208]))
    for index, bar_label in enumerate(barlabels):
        bar_label.set_x(bar_label.get_position()[1] - 200 + barpaddings[index])
    for i, text in enumerate(axs["Top"].texts):
        text.set_color(barcolors[i])
        text.set_size(20)
    axs["Top"].set_title("Regional Prob. that drought has decreased TFP (Barplot)", size=25, weight="bold")
    axs["Top"].set_xlabel("Probability (%)", weight="bold", size=20)
    axs["Top"].set_ylabel("Region", weight="bold", size=20)
    axs["Top"].set_yticklabels([])
    axs["Top"].xaxis.set_tick_params(labelsize=20)

    plt.savefig("figures/drought_fig3.png", bbox_inches='tight')

# plot figure 4
if plotFig4:

    fig, axis = plt.subplots(3,1, figsize=(12,12))

    axes = [axis[0], axis[1], axis[2]]
    thresholds = [None, .84, .16]
    threshold_labels = ["Maximum-Likelihood", "Upper-Bound", "Lower-Bound"]

    for threshold, label, axis in zip(thresholds, threshold_labels, axes):
        region_percent_loss = {}
        for region, samples in model_data["percent_loss_by_region"].items():
            if threshold != None:
                region_percent_loss[region] = np.quantile(samples, threshold)
            else:
                region_percent_loss[region] = np.mean(samples)
        
        data[f"region_percent_loss_{label}"] = list(map(lambda x : region_percent_loss[x], data.region23))
        
        region_geopandas = geopandas.read_file(
            geopandas.datasets.get_path('naturalearth_lowres')
        )
        
        region_geopandas = region_geopandas.merge(
            data,
            how='inner',
            left_on=['iso_a3'],
            right_on=['country']
        )
        bins = list(mapclassify.Quantiles(region_geopandas[f'region_percent_loss_Maximum-Likelihood'], k=10).bins)
        # delete bins that have similar range for map readability
        bins_to_remove = []
        for index, bin in enumerate(bins):
            if index != 0:
                if int(str(bin).split(".")[0]) == int(str(bins[index-1]).split(".")[0]):
                    bins_to_remove.append(index)
        bins = [val for index, val in enumerate(bins) if index not in bins_to_remove]
        legend_labels = []
        bins_as_int = [-1*int(str(bin).split(".")[0]) for bin in bins]
        for index, bin in enumerate(bins_as_int):
            if index == 0:
                legend_labels.append(f" > {bin} %")
            elif index == len(bins_as_int) - 1:
                legend_labels.append(f" < {bins_as_int[index-1]} %")
                break
            else:
                legend_labels.append(f"{bins_as_int[index-1]} % - {bin} %")
        region_geopandas.plot(column=f'region_percent_loss_{label}', cmap='RdYlGn', scheme="User_Defined", 
                legend=True, classification_kwds=dict(bins=bins), ax=axis)
        
        # data_ndcr = data[~data.country.isin(countries_with_no_drought)]
        # region_geopandas[region_geopandas.iso_a3.isin(countries_with_no_drought)].plot(color='gray', ax=axis)

        cmap = cm.get_cmap('RdYlGn')
        legend_bins = []
        for index, legend_label in enumerate(legend_labels):
            legend_bins.append(mpatches.Patch(color=cmap((1/len(legend_labels)*index)), label=legend_label))
        axis.legend(handles=legend_bins)
        
        axis.set_title(f"{label} Historical TFP Loss from Drought by region", size=15, weight="bold")
        axis.set_yticklabels([])
        axis.set_xticklabels([])

    plt.tight_layout()
    plt.savefig("figures/drought_fig4.png", bbox_inches='tight')

# plot figure 5
if plotFig5:

    fig, axis = plt.subplots(3,1, figsize=(12,6), layout="tight")
    _, bins, patches = axis[0].hist(model_data["percent_loss_by_region"]["Northern America"], bins=200, density=True)
    sns.kdeplot(model_data["percent_loss_by_region"]["Northern America"], ax=axis[0], color="black")

    mean = np.mean(model_data["percent_loss_by_region"]["Northern America"])
    std = np.std(model_data["percent_loss_by_region"]["Northern America"])

    print("North America")
    print("Mean:", mean)
    print("Upper bound:", mean + std)
    print("Lower bound:", mean-std)
    print("Percent samples below 0:", len([val for val in model_data["percent_loss_by_region"]["Northern America"] if val < 0])/len(model_data["percent_loss_by_region"]["Northern America"]))

    axis[0].axvline(mean, color = "yellow", linewidth=4)
    axis[0].axvline(mean - std, color = "orange", linewidth=4)
    axis[0].axvline(mean + std, color = "orange", linewidth=4)
    axis[0].axvline(0, color = "red", linewidth=4)
    for index, bin in enumerate(bins):
        if index != len(bins)-1:
            if bin < mean - std or bin > mean + std:
                patches[index-1].set_facecolor("gray")
            else:
                patches[index-1].set_facecolor("blue")

    mean = np.mean(model_data["percent_loss_by_region"]["Eastern Africa"])
    std = np.std(model_data["percent_loss_by_region"]["Eastern Africa"])

    print("Eastern Africa")
    print("Mean:", mean)
    print("Upper bound:", mean + std)
    print("Lower bound:", mean-std)
    print("Percent samples below 0:", len([val for val in model_data["percent_loss_by_region"]["Eastern Africa"] if val < 0])/len(model_data["percent_loss_by_region"]["Eastern Africa"]))

    _, bins, patches = axis[1].hist(model_data["percent_loss_by_region"]["Eastern Africa"], bins=200, density=True)
    sns.kdeplot(model_data["percent_loss_by_region"]["Eastern Africa"], ax=axis[1], color="black")
    axis[1].axvline(mean, color = "yellow", linewidth=4)
    axis[1].axvline(mean - std, color = "orange", linewidth=4)
    axis[1].axvline(mean + std, color = "orange", linewidth=4)
    axis[1].axvline(0, color = "red", linewidth=4)
    for index, bin in enumerate(bins):
        if index != len(bins)-1:
            if bin < mean - std or bin > mean + std:
                patches[index].set_facecolor("gray")
            else:
                patches[index].set_facecolor("blue")

    mean = np.mean(model_data["percent_loss_by_region"]["Southern Asia"])
    std = np.std(model_data["percent_loss_by_region"]["Southern Asia"])

    print("Southern Asia")
    print("Mean:", mean)
    print("Upper bound:", mean + std)
    print("Lower bound:", mean-std)
    print("Percent samples below 0:", len([val for val in model_data["percent_loss_by_region"]["Southern Asia"] if val < 0])/len(model_data["percent_loss_by_region"]["Southern Asia"]))

    _, bins, patches = axis[2].hist(model_data["percent_loss_by_region"]["Southern Asia"], bins=200, density=True)
    sns.kdeplot(model_data["percent_loss_by_region"]["Southern Asia"], ax=axis[2], color="black")
    axis[2].axvline(mean, color = "yellow", linewidth=4)
    axis[2].axvline(mean - std, color = "orange", linewidth=4)
    axis[2].axvline(mean + std, color = "orange", linewidth=4)
    axis[2].axvline(0, color = "red", linewidth=4)
    for index, bin in enumerate(bins):
        if index != len(bins)-1:
            if bin < mean - std or bin > mean + std:
                patches[index].set_facecolor("gray")
            else:
                patches[index].set_facecolor("blue")

    axis[0].set_xlim([-50,50])
    axis[1].set_xlim([-50,50])
    axis[2].set_xlim([-50,50])

    axis[0].set_title("Northern America", size=15, weight="bold")
    axis[1].set_title("Eastern Africa", size=15, weight="bold")
    axis[2].set_title("Southern Asia", size=15, weight="bold")

    axis[0].set_ylabel("")
    axis[1].set_ylabel("")
    axis[2].set_ylabel("")

    axis[1].set_title("Eastern Africa", size=15, weight="bold")
    axis[2].set_title("Southern Asia", size=15, weight="bold")

    fig.supxlabel("% Historical TFP Change from Drought", weight="bold", size=15)
    fig.supylabel("Probability Density", weight="bold", size=15)

    plt.savefig("figures/drought_fig5.png", bbox_inches='tight')

# plot supplementary figure 1
if plotSupFig1:

    q1 = np.quantile(model_data["global_percent_loss"], .16)
    q2 = np.quantile(model_data["global_percent_loss"], .84)
    fig, ax = plt.subplots()
    _, bins, patches = ax.hist(model_data["global_percent_loss"], bins=100, density=True)
    sns.kdeplot(model_data["global_percent_loss"], ax=ax, color="black")
    ax.axvline(q1, color = "blue", linewidth=4)
    ax.axvline(q2, color = "blue", linewidth=4)
    ax.axvline(0, color = "orange", linewidth=4)
    ax.axvline(np.mean(model_data["global_percent_loss"]), color = "green", linewidth=4)
    for index, bin in enumerate(bins):
        if index != len(bins)-1:
            if bin < q1 or bin > q2:
                patches[index].set_facecolor("gray")
            else:
                patches[index].set_facecolor("red")

    plt.xlabel("% Global TFP Change due to Drought", size=15, weight="bold")
    plt.ylabel("Probability Density", size=15, weight="bold")
    plt.savefig("figures/drought_supfig1.png", bbox_inches='tight')

    print("Percent samples below 0:", len([val for val in model_data["global_percent_loss"] if val < 0])/len(model_data["global_percent_loss"]))
    print("Mean:", np.mean(model_data["global_percent_loss"]))
    print("Q1:", q1)
    print("Q2:", q2)

# plot supplementary figure 2
if plotSupFig2:

    fig, axis = plt.subplots()

    development_percent_loss = {}
    dpl_q1 = {}
    dpl_q2 = {}
    for development, samples in model_data["development_percent_loss"].items():
        development_percent_loss[development] = np.mean(samples)
        dpl_q1[development] = np.quantile(samples, .16)
        dpl_q2[development] = np.quantile(samples, .84)
    data_mod["development_percent_loss"] = list(map(lambda x : development_percent_loss[x], data_mod.development))

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    country_geopandas = country_geopandas.merge(
        data_mod,
        how='inner', 
        left_on=['iso_a3'],
        right_on=['country']
    )

    country_geopandas.plot(column="development_percent_loss", cmap='summer', ax=axis)

    legend_labels = [f"{development}: Mean: {round(development_percent_loss[development], 1)}%, CI (1 SD): [{round(dpl_q1[development], 1)}% , {round(dpl_q2[development], 1)}]" for development in ["Developed","Developing"]]
    color_labels = [0,1000]
    legend_bins = []
    cmap = cm.get_cmap('summer')
    for color, legend_label in zip(color_labels, legend_labels):
        legend_bins.append(mpatches.Patch(color=cmap(color), label=legend_label))
    fig.legend(handles=legend_bins, loc="outside lower center")

    axis.set_title(f"Historical TFP Loss from Drought by Development", size=12, weight="bold")
    axis.set_yticklabels([])
    axis.set_xticklabels([])

    fig.subplots_adjust(bottom=-.15)

    plt.savefig("figures/drought_supfig2.png", bbox_inches='tight')

# plot supplementary figure 3
if plotSupFig3:

    fig, axes = plt.subplots(2,1)

    country_geopandas = geopandas.read_file(
            geopandas.datasets.get_path('naturalearth_lowres')
        )

    country_geopandas = country_geopandas.merge(
        data,
        how='inner', 
        left_on=['iso_a3'],
        right_on=['country']
    )
    plot1 = country_geopandas.plot(column="global_ag_weights", cmap="coolwarm", ax=axes[0],  legend=True, legend_kwds={"location":"bottom","pad":.04, "shrink": 0.6})
    plot1.figure.axes[-1].tick_params(labelsize=10)
    plot2 = country_geopandas.plot(column="regional_ag_weights", cmap="coolwarm", ax=axes[1],  legend=True, legend_kwds={"location":"bottom","pad":.04, "shrink": 0.6})
    plot2.figure.axes[-1].tick_params(labelsize=10)

    axes[0].set_title("% of Agricultural Revenue by Country (Global)", size=10, weight="bold")
    axes[1].set_title("% of Agricultural Revenue by Country (Regional)", size=10, weight="bold")

    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])

    fig.tight_layout()
    plt.savefig("figures/drought_supfig3.png", bbox_inches='tight')

# generate supplementary table 1

if genSupTab1:
    df_tab1 = pd.DataFrame()
    df_tab1["ISO3 Country Code"] = sorted(set(data.country))    
    df_tab1[f"{model.capitalize()} Drought Coef. Mean"] = list(map(lambda x : np.mean(model_data["country_coefficients"][x+"_country_fixed_effect"]), df_tab1["ISO3 Country Code"]))
    df_tab1[f"{model.capitalize()} Drought Coef. SD"] = list(map(lambda x : np.std(model_data["country_coefficients"][x+"_country_fixed_effect"]), df_tab1["ISO3 Country Code"]))
    df_tab1["\% Drought Coef Samples $<$ 0"] = list(map(lambda x : model_data["country_percentiles"][x]*100, df_tab1["ISO3 Country Code"]))
    df_tab1["Historical \% TFP Change from Drought Mean"] = list(map(lambda x : np.mean(model_data["percent_loss_by_country"][x]), df_tab1["ISO3 Country Code"]))
    df_tab1["Historical \% TFP Change from Drought SD"] = list(map(lambda x : np.std(model_data["percent_loss_by_country"][x]), df_tab1["ISO3 Country Code"]))
    df_tab1.to_latex(buf="tables/suptab1.ltx", index=False, float_format="%.3f", longtable=True)

# generate supplementary table 2

if genSupTab2:
    df_tab2 = pd.DataFrame()
    df_tab2["Region Name"] = sorted(set(data.region23))    
    df_tab2["\% Region Impact Samples $<$ 0"] = list(map(lambda x : model_data["region_percentiles"][x]*100, df_tab2["Region Name"]))
    df_tab2["Historical \% TFP Change from Drought Mean"] = list(map(lambda x : np.mean(model_data["percent_loss_by_region"][x]), df_tab2["Region Name"]))
    df_tab2["Historical \% TFP Change from Drought SD"] = list(map(lambda x : np.std(model_data["percent_loss_by_region"][x]), df_tab2["Region Name"]))
    df_tab2.to_latex(buf="tables/suptab2.ltx", index=False, float_format="%.3f")

# generate supplementary table 3

if genSupTab3:

    ndvi_model = pd.read_pickle("output/models/bayes_models/ndvi_bayes_yfe_cre_for_drought_full/ndvi_bayes_yfe_cre_for_drought_full_with_intercept_temp_multiyear_drought_only_country_coefs.pkl")
    ndvi_data = pd.read_csv("data/regression/ndvi_regression_data.csv").dropna().reset_index(drop=True)

    # unscale ndvi country coefficients
    scaled_vars = {}
    unscaled_vars = {}
    for country_index, var in enumerate(ndvi_model["var_list"][43:]):
        scaled_vars[var] = ndvi_model["posterior"][:,:,:,country_index].data.flatten()
    for var, samples in scaled_vars.items():
        unscaled_vars[var] = np.array(samples) * np.std(ndvi_data.fd_ln_ndvi)

    df_tab3 = pd.DataFrame()
    df_tab3["ISO3 Country Code"] = sorted(set(data.country))
    for model in model_impacts:
        df_tab3[f"{model.capitalize()} Drought Coef. Means"] = list(map(lambda x : np.mean(model_impacts[model]["country_coefficients"][x+"_country_fixed_effect"]), df_tab3["ISO3 Country Code"]))
    df_tab3["Model3 Drought Coef. Means"] = list(map(lambda x : np.mean(unscaled_vars[x+"_country_fixed_effect"] if x+"_country_fixed_effect" in unscaled_vars else np.NaN), df_tab3["ISO3 Country Code"]))
    df_tab3.to_latex(buf="tables/suptab3.ltx", index=False, float_format="%.3f", longtable=True)
