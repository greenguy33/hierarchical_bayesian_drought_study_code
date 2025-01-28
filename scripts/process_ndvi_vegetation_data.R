# setwd("hierarchical_bayesian_drought_study_code/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")

country_shapes = read_sf("country_shapes",layer="country")

dir <- "../data/PKU_GIMMS_NDVI_AVHRR_MODIS/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated/"
files <- list.files(path=dir)

lapply(files, function(file) {

  year = strsplit(strsplit(file, split = "_")[[1]][5], split="\\.")[[1]][1]
  
  vegetation_raster = stack(str_interp("${dir}/${file}"))
  vegetation_raster[vegetation_raster == 65535] <- 0

  extracted_data = exact_extract(vegetation_raster, country_shapes, fun = "mean")

  colnames(extracted_data) <- c("raw_ndvi","qc_layer")

  data <- c()
  data$country <- country_shapes$FIPS_CNTRY
  data$ndvi <- extracted_data[1]
  write.csv(data, str_interp("../data/PKU_GIMMS_NDVI_AVHRR_MODIS/extracted/vegetation_coverage.${year}.csv"))

})
