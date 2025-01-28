import pandas as pd
import os
import numpy as np
import country_converter as cc

directory = "../data/PKU_GIMMS_NDVI_AVHRR_MODIS/extracted/"
ndvi_files = os.listdir(directory)

green_months = {}
green_month_data = pd.read_csv("../data/peak_bottom_ndvi_month_country.csv")
for row in green_month_data.itertuples():
    peak = row[5]
    season = [i for i in range(peak-2,peak+3)]
    for i in range(len(season)):
        if season[i] < 1:
            season[i] = season[i] + 12
        elif season[i] > 12:
            season[i] = season[i] - 12
    green_months[row.ISO3] = season

code_conversion = {}
for country in set(pd.read_csv(directory + "vegetation_coverage.19820101.csv")["country"]):
    code_conversion[country] = cc.convert(names=[country], to="ISO3", not_found=None)

# drop some iso2 codes that map to the same iso3 code as another iso2 code
codes_to_drop = ["UK"]

yearly_mean_dataframes = []
for year in range(1982, 2023):
    yearly_data = [pd.read_csv(directory + file) for file in ndvi_files if str(year) in file]
    yearly_data_files = [file for file in ndvi_files if str(year) in file]
    country_means = {}
    for i, df in enumerate(yearly_data):
        yearly_data[i] = yearly_data[i].drop(yearly_data[i][yearly_data[i]["country"].isin(codes_to_drop)].index).reset_index(drop=True)
        yearly_data[i]["country"] = [code_conversion[country] for country in yearly_data[i]["country"]]
    for country in yearly_data[0]["country"]:
        if country in green_months:
            country_means[country] = []
            datasets_to_use = [index for index, file in enumerate(yearly_data_files) if int(file.split(".")[-2].split(str(year))[1][0:2].replace("0","")) in green_months[country]]
            for dataset in datasets_to_use:
                country_means[country].append(yearly_data[dataset].loc[yearly_data[dataset]["country"]==country]["raw_ndvi"].item())
            country_means[country] = np.mean(country_means[country]) / 1000
    
    yearly_mean_data = pd.DataFrame()
    yearly_mean_data["country"] = list(country_means.keys())
    yearly_mean_data["year"] = [year] * len(country_means)
    yearly_mean_data["ndvi"] = list(country_means.values())
    yearly_mean_dataframes.append(yearly_mean_data)

pd.concat(yearly_mean_dataframes).replace(0,np.NaN).sort_values(["country","year"]).to_csv("../data/PKU_GIMMS_NDVI_AVHRR_MODIS/pku_data_aggregated.csv")