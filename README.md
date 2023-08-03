# AI-Seafog-forecast

## Brief Description

Accurate and timely prediction of sea fog is very important for activities related to maritime and coastal economies. However, traditional methods based on numerical and statistical methods often struggle because sea fog is complex and unpredictable. In this study, we aimed to develop a smart machine learning method to predict sea fog. We focused on the Yangtze River Estuary (YRE) coastal area as an example.  We present a novel approach for predicting sea fog by utilizing the output from a numerical weather prediction (NWP) model. We used a technique named time-lagged correlation analysis (TLCA) to determine the important factors causing sea fog. We also used ensemble learning and a special method named the focal loss function to handle situations where there is an imbalance between the fog and fog-free data. To test how accurate our method is, we used data from weather stations and historical forecasts over one year. Our machine learning approach performed better than two other methods—the weather research and forecasting nonhydrostatic mesoscale model (WRF-NMM) and the method developed by the National Oceanic and Atmospheric Administration (NOAA) Forecast Systems Laboratory (FSL)—in predicting sea fog. Specifically, regarding predicting sea fog 60 hours in advance with a visibility of 1 kilometer or less, our method achieved better results by increasing the chances of detecting sea fog while reducing false alarms.

## Data

The data used in this study can be found at [].

- yyyy_spot[00/12]_wrf61_prev18.csv: 

This dataset includes visibility observations, station latitude and longitude, time and model forecast variables from 2014 to 2020.

The visibility observations were obtained from the Meteorological Information Comprehensive Analysis and Process System (MICAPS), which provided various weather measurements including temperature, pressure, humidity, precipitation, visibility, current weather, and other meteorological variables. Meteorological measurements were recorded at 3-hour intervals. Besides, the location and time of each record were provided.

The model forecast variables are from the Weather Research and Forecasting Nonhydrostatic Mesoscale Model (WRF-NMM) provided by the US National Weather Service (NCEP). The WRF-NMM produces hourly predictions for a 60-hour forecast period, launching twice daily at 00:00 UTC and 12:00 UTC. Forecast variables incorporated in the model encompass temperature, relative humidity, cloud cover, wind speed, precipitation, and more. This dataset includes 26 variables, which are listed below.

| Variables	| Long Name	| Description |
|  ----  |  ----  |  ----  |
| DPT_GDS3_HTGL |	Dew point   |  Height in 2 meters |
| HGT_GDS3_0DEG |	Geopotential  | height	Level of 0 deg (C) isotherm |
| HGT_GDS3_CEIL |	Geopotential  | height	Cloud ceiling |
| HGT_GDS3_HTFL |	Geopotential  | height	Highest tropospheric freezing level |
| HGT_GDS3_SFC |	Geopotential   | height	Ground/water surface |
| T_CDC_GDS3_EATM |	Total cloud cover  |	Entire atmosphere |
| H_CDC_GDS3_HCY |	High cloud cover |	High cloud layer |
| M_CDC_GDS3_MCY |	Mid cloud cover	 | Mid cloud layer |
| L_CDC_GDS3_MCY |	Low cloud cover	 | Low cloud layer |
| PLI_GDS3_SPDY |	Parcel lifted index (to 500 hPa)  |	Layer between two levels at specified pressure difference from ground to level |
| LFT_X_GDS3_ISBY |	Surface lifted index	 | Layer between two isobaric levels |
| PRES_GDS3_SFC |	Pressure	 | Ground/water surface |
| PRMSL_GDS3_MSL |	Pressure reduced to Mean sea level  |	Mean sea level |
| MSLET_GDS3_MSL |	Mean sea level pressure	 | Mean sea level |
| P_WAT_GDS3_EATM |	Precipitable water	 | Entire atmosphere |
| POP_GDS3_SFC |	Probability of precipitation	 | Ground/water surface |
| R_H_GDS3_HTGL |	Relative humidity	 | Height in 2 meters |
| R_H_GDS3_HYBL |	Relative humidity	 | Hybrid level |
| SPF_H_GDS3_HTGL |	Specific humidity	 | Height in 2 meters |
| SPF_H_GDS3_SPDY |	Specific humidity	 | Layer between two levels at specified pressure difference from ground to level |
| TMP_GDS3_HTGL |	Temperature	 | Height in 2 meters |
| TMP_GDS3_SFC |	Temperature	 | Ground/water surface |
| U_GRD_GDS3_ HTGL |	U-component of wind	 | Height in 10 meters |
| U_GRD_GDS3_SPDY |	U-component of wind	 | Layer between two levels at specified pressure difference from ground to level |
| V_GRD_GDS3_HTGL |	V-component of wind	 | Height in 10 meters |
| V_GRD_GDS3_SPDY |	V-component of wind	 | Layer between two levels at specified pressure difference from ground to level |

 
To integrate the gridded model forecast data with sparse observation data, we have a two-step pre-processing process. Firstly, we synchronized the time zones of two datasets to UTC+00:00. Secondly, we used the Inverse Distance Weighted (IDW) approach for spatial matching. As a result, we generated a multi-variable time series dataset as follows: 

$$D=\{X_{N \times M \times T}, Y_{N \times T}\}$$  

where $X$ is a three-dimensional tensor of size $N\times M\times T$ (corresponding to $N$ samples of $M$ forecast variables of $T$ hours), and $Y$ is the observational data of size $N\times T$ (corresponding to $N$ samples of $T$ hours). A sample from dataset D is represented as $s= \{ x_{M \times T}, y_T \} \in D$. In this datasets, $M=27$, and $T=60$. Besides, this dataset includes 18 hours of visibility observations before the model launching. It should be noted that some observations are missing for some reason.


- yyyy_spot[00/12]_wrf6_prev18.csv: 

This datasets are from yyyy_spot00/12_wrf61_prev18.csv and is a sliding-window extraction of 61 forecast moments, with each sliding-window consisting of 6 hours.

## Time-lagged Correlation Analysis (TLCA)

The details of the TLCA algorithm are described below.

1. We set the maximum lag time $\tau$ and select a subdataset from  $D$, which is denoted as $D_{\mathrm{TCCA}}:D_{\mathrm{TCCA}}=\{X_t,y_0\}\subseteq D,=0,-1,-2,\ldots,-\tau$. The observed visibility is denoted as $y_0$. The forecast variables before the observation time are denoted as $X_t$. In this work, we set $\tau=5$ because of the trade-off between the number of predictors and forecast accuracy.
2. By using the Pearson correlation coefficient, we determined how closely $y_0$ is related to each of the variables in $X_t$. The variables that do not pass the significance test were omitted from the analysis (the threshold for statistical significance was set to $\alpha=0.05$). 



## Forecasting Methods

As ground truth for model training, we separated sea fog events into two groups: fog (visibility $\le$ 1 km) and fog-free (visibility $>$ 1 km). The predictors of the forecasting model can be split into five categories: 
1. time-lagged predictors extracted using the TLCA method;
2. station location (in terms of latitude and longitude);
3. hour, day, and month;
4. visibility observations for the 6 hours preceding the WRF-NMM launch;
5. forecast lead time (from 1 to 60 hours).

We used LightGBM (Ke et al., 2017) to make forecasts for different types of sea fog. The focal loss (Lin et al., 2018) and ensemble learning (Galar et al., 2012) are used to improve the ML model's prediction abilities while addressing the issue of data imbalance between fog and fog-free events.

## How to use?

### Data Download

https://www.zenodo.org/badge/DOI/10.5281/zenodo.8204518.svg


You can find and download the data used in this study at [zenodo](https://doi.org/10.5281/zenodo.8204518.)

### Requirements

- numpy
- pandas
- lightgbm
- matplotlib
- seaborn
- cartopy
- imblearn
- scipy
- sklearn


## Reference
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree, in: Advances in Neural Information Processing Systems.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P., 2018. Focal Loss for Dense Object Detection.
- Galar, M., Fernandez, A., Barrenechea, E., Bustince, H., Herrera, F., 2012. A Review on Ensembles for the Class Imbalance Problem: Bagging-, Boosting-, and Hybrid-Based Approaches. IEEE Trans. Syst., Man, Cybern. C 42, 463–484. https://doi.org/10.1109/TSMCC.2011.2161285
















