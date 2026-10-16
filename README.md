# AirSight AI 🌍

**Air Quality Index Estimation Across Global Cities**

AirSight AI is a machine learning project that explores air pollution data from cities around the world and estimates the overall Air Quality Index (AQI) using pollutant AQI values.

I built this project to understand how machine learning can be applied to environmental data and how the results can be presented through a simple, interactive dashboard.

## What it does

- **Explore cities:** Browse the cities and countries available in the dataset.
- **Estimate AQI:** Enter pollutant AQI values to estimate the overall AQI using a trained machine learning model.
- **View pollution data:** Explore the available AQI and pollutant information through charts and summaries.
- **Compare model performance:** View evaluation metrics from the experiments conducted in Google Colab.
- **Upload a dataset:** Upload a compatible CSV file to analyse its data and generate predictions where the required columns are available.

## How it works

The project uses a global air pollution dataset and compares different regression models. Random Forest was selected for the current AQI estimation model based on its evaluation results.

The model uses these four inputs:

- CO AQI Value
- Ozone AQI Value
- NO2 AQI Value
- PM2.5 AQI Value

The trained model was developed and evaluated in Google Colab. The dashboard is built with Python and Streamlit.

## Model evaluation

The current model produced the following results during testing:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.232 |
| Root Mean Squared Error (RMSE) | 2.662 |
| R² Score | 0.9978 |
| Predictions within 5 AQI points | 99.28% |

Five-fold cross-validation produced a mean MAE of 0.1984 and a mean R² of 0.9982.

These results apply to the current estimation task. Since the overall AQI is closely related to pollutant AQI sub-indices, the scores should not be interpreted as evidence of future forecasting accuracy.

## Tech stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Google Colab

## Dataset

The project uses a global air pollution dataset containing city and country information, overall AQI values, and pollutant-specific AQI values.

## Current limitations

The dataset used for this version does not contain dated observations. The current model estimates overall AQI from pollutant AQI inputs; it does not predict how AQI will change on a future date.

Future improvements could include using dated air quality observations, adding weather information, and developing a model specifically for forecasting AQI over time.

---

