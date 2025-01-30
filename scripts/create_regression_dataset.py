import csv
import pandas as pd
import numpy as np
from countrycode import countrycode as cc
import warnings
from sklearn.preprocessing import OrdinalEncoder
import itertools as it
from bisect import bisect_left

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def format_tfp_data(data, years, year_column_format, country_column, output_var):
	formatted_outcome_var = {}
	for row in data.iterrows():
		row = row[1]
		country = row[country_column]
		if country in iso3_countries_with_climate_data:
			formatted_outcome_var[country] = {}
			for year in years:
				year_data = float(row[year_column_format.replace("year",str(year))])
				last_year_data = float(row[year_column_format.replace("year",str(year-1))])
				formatted_outcome_var[country][year] = {}
				if np.isnan(year_data) or np.isnan(last_year_data):
					formatted_outcome_var[country][year][output_var] = np.NaN
				else:
					outcome = np.log(float(year_data)) - np.log(float(last_year_data))
					formatted_outcome_var[country][year][output_var] = outcome
	return formatted_outcome_var


def format_ndvi_data(data, years, year_column, country_column, output_var):
	data["ln_ndvi"] = np.log(data["ndvi"])
	data[output_var] = data.groupby(country_column)["ln_ndvi"].diff()
	formatted_outcome_var = {}
	for row in data.iterrows():
		row = row[1]
		if row[country_column] in iso3_countries_with_climate_data and row[year_column] in years:
			if row[country_column] not in formatted_outcome_var: 
				formatted_outcome_var[row[country_column]] = {}
			if row[year_column] not in formatted_outcome_var[row[country_column]]:
				formatted_outcome_var[row[country_column]][row[year_column]] = {}
			formatted_outcome_var[row[country_column]][row[year_column]][output_var] = row[output_var]
	return formatted_outcome_var


def add_climate_vars_to_dataset(dataset, climate_val, prev_climate_val, country, year, climate_var, weights):
	dataset[country][year][f"{climate_var}_{weights}"] = climate_val
	dataset[country][year][f"{climate_var}_{weights}_2"] = np.square(climate_val)
	dataset[country][year][f"{climate_var}_{weights}_3"] = np.power(climate_val,3)
	if prev_climate_val != None:
		dataset[country][year][f"fd_{climate_var}_{weights}"] = climate_val - prev_climate_val
		dataset[country][year][f"fd_{climate_var}_{weights}_2"] = np.square(climate_val) - np.square(prev_climate_val)
		dataset[country][year][f"fd_{climate_var}_{weights}_3"] = np.power(climate_val,3) - np.power(prev_climate_val,3)
	return dataset

def add_fixed_effects_to_dataset(file):
	dataset = pd.read_csv(file)
	dataset["region23"] = cc(dataset.country, origin="iso3c", destination="region23").replace(" ","_")
	for country in sorted(list(set(dataset.country))):
		dataset[f"{country}_country_fixed_effect"] = np.where(dataset.country == country, 1, 0)
	for year in sorted(list(set(dataset.year))):
		dataset[f"{year}_year_fixed_effect"] = np.where(dataset.year == year, 1, 0)
	dataset.to_csv(file)

def write_regression_data_to_file(file, data, first_year):
	writer = csv.writer(file)
	headers =["country","year"]
	for column in data["AFG"][first_year]:
		headers.append(column.replace(" ","_").lower())
	writer.writerow(headers)
	for country, data_by_year in dict(sorted(data.items(), key=lambda x: x[0])).items():
		for year, data in data_by_year.items():
			new_row = [country,year]
			for column in data:
				new_row.append(data[column])
			writer.writerow(new_row)

def find_closest_to_value_in_list(list_of_values, value_to_position, ):
	position = bisect_left(list_of_values, value_to_position)
	if position == 0:
		return position
	if position == len(list_of_values):
		return -1
	before = list_of_values[position - 1]
	after = list_of_values[position]
	if after - value_to_position < value_to_position - before:
		return position
	else:
		return position - 1
	
def add_natural_disasters_to_dataset(dataset, extracted_disasters, countries_with_natural_disaster_data):
	disaster_combinations = list(it.combinations(disaster_types_to_extract, 2))
	for country, data_by_year in dataset.items():
			for year, data in data_by_year.items():
				if country not in countries_with_natural_disaster_data:
					for disaster in disaster_types_to_extract:
						dataset[country][year][disaster] = np.NaN
					for disaster_combination in disaster_combinations:
						dataset[country][year][disaster_combination[0] + "_" + disaster_combination[1]] = np.NaN
				else:
					disaster_data = {}
					for disaster in disaster_types_to_extract:
						if country in extracted_disasters and year in extracted_disasters[country] and disaster in extracted_disasters[country][year]:
							disaster_data[disaster] = 1
						else:
							disaster_data[disaster] = 0
					for disaster_combination in disaster_combinations:
						if disaster_data[disaster_combination[0]] == 1 and disaster_data[disaster_combination[1]] == 1:
							disaster_data[disaster_combination[0] + "_" + disaster_combination[1]] = 1
						else:
							disaster_data[disaster_combination[0] + "_" + disaster_combination[1]] = 0
					for disaster, value in disaster_data.items():
						dataset[country][year][disaster] = value
	return dataset

tfp_data = pd.read_csv("data/TFP/AgTFPInternational2021_AG_TFP.csv", header=2)
ndvi_data = pd.read_csv("data/PKU_GIMMS_NDVI_AVHRR_MODIS/pku_data_aggregated.csv")
natural_disasters_data = pd.read_csv("data/natural_disasters/emdat_1960-2024.csv")
disaster_types_to_extract = ["Drought"]
tfp_years = range(1962,2022)
ndvi_years = range(1983,2023)

fips_countries_with_climate_data = list(pd.read_csv("data/temp/monthly/processed_by_country/unweighted/temp.monthly.bycountry.unweighted.mean.csv").country)
iso3_country_list = cc(fips_countries_with_climate_data, origin="fips", destination="iso3c")
iso3_countries_with_climate_data = []
fips_indices_to_delete = []
for index, country in enumerate(iso3_country_list):
	if iso3_country_list[index] != None:
		iso3_countries_with_climate_data.append(country)
	else:
		fips_indices_to_delete.append(index)
fips_indices_to_delete = sorted(fips_indices_to_delete, reverse=True)
for index in fips_indices_to_delete:
	del fips_countries_with_climate_data[index]
assert len(fips_countries_with_climate_data) == len(iso3_countries_with_climate_data)
fips_to_iso3_country_map = {
	fips_countries_with_climate_data[index]:iso3_countries_with_climate_data[index]
	for index in range(len(iso3_countries_with_climate_data))
}

print("Adding TFP data...")
formatted_tfp_data = format_tfp_data(tfp_data, tfp_years, "year", "ISO3", "fd_ln_tfp")

print("Adding NDVI data...")
formatted_ndvi_data = format_ndvi_data(ndvi_data, ndvi_years, "year", "country", "fd_ln_ndvi")

print("Adding natural disaster data...")

extracted_disasters = {}
missing_countries = set()
extreme_temp_subtypes = set()
countries_with_natural_disaster_data = set(natural_disasters_data.ISO)
for row in natural_disasters_data.iterrows():
	row = row[1]
	disaster_type = row["Disaster Type"]
	disaster_subtype = row["Disaster Subtype"]
	disaster = None
	if disaster_type in disaster_types_to_extract:
		disaster = disaster_type
	elif disaster_subtype in disaster_types_to_extract:
		disaster = disaster_subtype
	if disaster != None:
		country = row.ISO
		year = int(row["DisNo."].split("-")[0])
		if country not in extracted_disasters:
			extracted_disasters[country] = {}
		if year not in extracted_disasters[country]:
			extracted_disasters[country][year] = {}
		extracted_disasters[country][year][disaster] = 1

formatted_tfp_data = add_natural_disasters_to_dataset(formatted_tfp_data, extracted_disasters, countries_with_natural_disaster_data)
formatted_ndvi_data = add_natural_disasters_to_dataset(formatted_ndvi_data, extracted_disasters, countries_with_natural_disaster_data)

print("Adding monthly climate data...")

for climate_var in ["temp","precip"]:
	for weights in ["unweighted", "pop_weighted","ag_weighted"]:
		aggregate_var = "mean"
		if weights != "unweighted":
			aggregate_var = "weighted_mean"
		weights_no_dash = weights.replace("_","")
		data = pd.read_csv(f"data/{climate_var}/monthly/processed_by_country/{weights}/{climate_var}.monthly.bycountry.{weights_no_dash}.mean.csv")
		for row in data.iterrows():
			prev_climate_val = None
			row = row[1]
			if row.country in fips_to_iso3_country_map:
				country = fips_to_iso3_country_map[row.country]
				for year in range(1960,2024):
					monthly_climate_vals = []
					for month in range(1,13):
						if month < 10:
							month = "0" + str(month)
						monthly_climate_vals.append(row[f"{weights_no_dash}_by_country.{aggregate_var}.X{year}.{month}.01"])
					annual_climate_mean = np.mean(monthly_climate_vals)
					if climate_var == "temp":
						# celsius to kelvin
						annual_climate_mean = annual_climate_mean - 273.15
					elif climate_var == "precip":
						# precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
						annual_climate_mean = annual_climate_mean * 2.628e+6
					if year in tfp_years and country in formatted_tfp_data:
						formatted_tfp_data = add_climate_vars_to_dataset(formatted_tfp_data, annual_climate_mean, prev_climate_val, country, year, climate_var, weights)
					if year in ndvi_years and country in formatted_ndvi_data:
						formatted_ndvi_data = add_climate_vars_to_dataset(formatted_ndvi_data, annual_climate_mean, prev_climate_val, country, year, climate_var, weights)
					prev_climate_val = annual_climate_mean

with open("data/regression/tfp_regression_data.csv", "w") as tfp_file:
	 write_regression_data_to_file(tfp_file, formatted_tfp_data, 1962)
with open("data/regression/ndvi_regression_data.csv", "w") as ndvi_file:
	 write_regression_data_to_file(ndvi_file, formatted_ndvi_data, 1983)

print("Adding fixed effects...")

add_fixed_effects_to_dataset("data/regression/tfp_regression_data.csv")
add_fixed_effects_to_dataset("data/regression/ndvi_regression_data.csv")

print("Results written to data/regression")