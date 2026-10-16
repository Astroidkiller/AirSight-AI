# AirSight AI — Global AQI Intelligence & Estimation Platform 🛰️

AirSight AI is a modern, high-performance environmental intelligence and diagnostic web application built for environmental researchers, data scientists, policy analysts, and concerned citizens. It synthesizes an empirical benchmark dataset spanning **23,463 global monitoring stations** across 175 sovereign countries with a pre-trained **Random Forest Regressor** to deliver instant multi-pollutant sub-index synthesis, granular city scorecards, batch dataset validation, and transparent machine learning evaluation diagnostics.

---

## 🌟 Application Modules & Key Capabilities

### 1. 🌐 Overview & Global Intelligence
- **Empirical Scope & KPI Suite**: Real-time aggregation of 23,463 monitoring points, 175 sovereign nations, 23,462 unique cities, global median AQI, and $PM_{2.5}$ dominance (~88.8%).
- **Interactive Choropleth**: World map illustrating national mean AQI with hover tooltips covering station counts, median, and maximum observed AQI.
- **EPA Category Distribution**: Standardized donut and bar charts categorized strictly according to US-EPA 40 CFR Part 58 breakpoints.
- **Multi-Pollutant Correlation Analysis**: Pearson correlation matrix and statistical spread box plots for $CO$, $O_3$, $NO_2$, and $PM_{2.5}$.

### 2. 🏙️ Global City & Country Explorer
- **Dynamic Cascading Selectors**: Filter across all sovereign countries and cities directly populated from the real dataset.
- **Granular City Scorecard**: Compares recorded local AQI against both national averages and global benchmarks.
- **Multi-Record Handling**: If a location has multiple monitoring station records, displays all records in an interactive table and calculates summary statistics (mean, min, max, dominant pollutant).
- **Sub-Index Profile**: Visualizes recorded sub-indices and EPA tiers ($CO$, $O_3$, $NO_2$, $PM_{2.5}$).
- **National Rankings**: Top 5 cleanest and top 5 most polluted cities in the chosen country.
- **Instant Simulator Action**: Transfer any city's readings directly into the ML Estimator with one click.

### 3. 🔬 ML-Based AQI Estimator & Diagnostic Engine
- **Sub-Index Synthesis**: Estimates overall continuous AQI from four input sub-indices in strictly enforced feature order:
  1. `CO AQI Value`
  2. `Ozone AQI Value`
  3. `NO2 AQI Value`
  4. `PM2.5 AQI Value`
- **Rigorous Input Validation**: Synchronized numeric inputs and sliders with boundary enforcement (rejects negative or missing values).
- **US-EPA Standardized Classification**: Displays official EPA health categories (Good, Moderate, USG, Unhealthy, Very Unhealthy, Hazardous) with actionable advisories for sensitive groups and the general public.
- **Governing Pollutant Diagnostics**: Identifies the primary driving pollutant behind composite AQI calculation.
- **Diagnostic Presets**: One-click quick presets for Pristine Air, Urban Moderate, Industrial Smog, and Wildfire Inversion.

### 4. 📊 Model Performance & Scientific Transparency
- **Recorded Colab Evaluation Benchmarks**:
  - Test MAE: **0.2318** pts
  - Test RMSE: **2.662** pts
  - Test $R^2$: **0.9978**
  - Accuracy within $\pm 5$ AQI points: **99.28%**
  - 5-Fold Cross-Validation Mean MAE: **0.1984**
  - 5-Fold Cross-Validation MAE Std Dev: **± 0.0401**
  - 5-Fold Cross-Validation Mean $R^2$: **0.9982**
- **Live Empirical Ground-Truth Validation**: Evaluated live across all 23,463 records with full transparency.
- **Gini Feature Importances**: Extracted directly from the fitted Random Forest ($PM_{2.5} \approx 98.07\%$, $Ozone \approx 1.80\%$, $CO \approx 0.08\%$, $NO_2 \approx 0.05\%$).
- **Parity & Residual Plots**: Actual vs Predicted AQI scatter with 1:1 diagonal reference line and residual error distribution histogram.

### 5. 📁 Custom Dataset Batch Upload & Validation
- **Custom CSV Ingestion**: File preview displaying rows, columns, and sample entries.
- **Schema Validation**: Flexible case-insensitive and whitespace-tolerant matching for the 4 required features.
- **Sample Template Download**: Downloadable pre-configured CSV template for testing.
- **Ground-Truth Evaluation**: If an `AQI Value` column is present in the uploaded file, the system automatically computes Batch MAE, RMSE, and $R^2$, and renders an uploaded parity plot.
- **Batch Export**: Export prediction results with one click.

### 6. ⏳ Date & Forecasting Transparency
- **Scientific Demarcation**: Clarifies that the cross-sectional dataset contains no timestamps and the Random Forest model is a **sub-index diagnostic synthesizer**, not a chronological time-series forecaster.
- **Scenario Planning Note**: Explains why selecting a date acts purely as an annotation and does not affect mathematical sub-index estimation.
- **Strict Labeling**: Uses honest designations like `AQI Estimate` and `Recorded Dataset Information`.

---

## 🛠️ Technology Stack & Dependencies

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Framework**: Streamlit
- **Machine Learning**: Scikit-learn, Joblib
- **Data Manipulation**: Pandas, NumPy
- **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
- **Excel Support**: Openpyxl

---

## 🚀 Quickstart & Local Launch

### 1. Clone or Open the Repository
```bash
cd AirSight_AI
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python -m streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
AirSight_AI/
├── .streamlit/
│   └── config.toml                        # Theme configuration (Navy & Teal palette)
├── aqi_model (3).pkl                      # Trained Random Forest model bundle
├── global_air_pollution_dataset[1].csv    # Benchmark dataset (23,463 records)
├── app.py                                 # Main production Streamlit application
├── requirements.txt                       # Project dependencies
├── Requirements.txt                       # Synchronized requirements file
└── README.md                              # Application documentation
```
