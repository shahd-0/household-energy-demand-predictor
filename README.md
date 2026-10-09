# household-energy-demand-predictor
A machine learning project using a Random Forest model trained on 2M+ smart-meter readings, deployed as a Flask REST service with a web interface and feedback loop. 

This project covers the full ML workflow, from data cleaning and preprocessing to a pilot study comparing several regression models, to finally deploying it as a client-server application.

## Features
- Cleans and explores the UCI Individual Household Electric Power Consumption dataset (about 2 million minute-level readings)
- Pilot study comparing Linear Regression, Decision Tree, and Random Forest regressors, plus a Logistic Regression classifier for Low / Medium / High demand
- Random Forest model trained and saved with `joblib`
- Flask REST service with input validation (`/predict`, `/feedback`)
- Browser interface with dropdowns and range-checked inputs, plus a prediction history stored in the browser
- Feedback endpoint that stores user-reported actual values in a CSV file for future retraining

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Flask, Flask-CORS, joblib, HTML/CSS/JavaScript

## Project structure

```
household-energy-demand-predictor/
├── backend/
│   ├── app.py            
│   └── train.py         
├── frontend/
│   └── index.html       
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_pilot_study.ipynb
├── data/
│   └── README.md       
├── requirements.txt
└── README.md
```

## Dataset

Source: [UCI Machine Learning Repository, Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

The dataset contains 2,075,259 minute-level measurements from a single household in France, collected over 47 months (December 2006 to November 2010). The raw and cleaned files are not stored in this repository because of their size. Download the raw file into `data/` and run the preprocessing notebook to reproduce the cleaned dataset.

| Attribute | Description | Unit |
|---|---|---|
| Global_active_power | Total active power consumed by the household per minute (target) | kilowatts |
| Global_reactive_power | Total reactive power consumed by the household per minute | kilowatts |
| Voltage | Average voltage in the household | volts |
| Global_intensity | Average current intensity in the household | amperes |
| Sub_metering_1 | Energy used in the kitchen | watt-hours |
| Sub_metering_2 | Energy used in the laundry room | watt-hours |
| Sub_metering_3 | Energy used by water heater and air conditioner | watt-hours |

## Data preparation

- Replaced `?` placeholders with missing values and converted columns to numeric types
- Dropped rows with missing values (about 1.25% of the data, spread across all numeric columns because the sensors record simultaneously). 2,049,280 rows remain
- Kept repeated readings, since they belong to different timestamps and are valid records
- Combined date and time into one datetime column and extracted hour, day, and month
- Explored distributions (the target is slightly right-skewed, ranging from 0.076 to 11.122 kW) and correlations between features
- Tree-based models do not need feature scaling, so they were trained on unscaled features
  
## Pilot study results

Regression on `Global_active_power` (80/20 random split, `random_state=42`):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.0257 | 0.0403 | 0.9986 |
| Decision Tree | 0.0156 | 0.0359 | 0.9989 |
| Random Forest (50 trees) | 0.0122 | 0.0259 | 0.9994 |

Random Forest performed best and was chosen for the final model.

## Final model

- **Model:** Random Forest Regressor (`n_estimators=200`, `max_depth=20`, `random_state=42`)
- **Features:** `hour`, `day_of_week`, `voltage`, `global_intensity`, `sub_metering_1`, `sub_metering_2`, `sub_metering_3`
- **Split:** 80/20 train/test

The trained model is saved with `joblib` as `model.pkl`, so the service loads it at startup without retraining.

## How it works

1. The web interface collects the inputs and sends them as JSON to the Flask service.
2. The service validates that every feature is present, numeric, and within its allowed range (for example, voltage must be between 200 and 260).
3. Valid input is passed to the model, and the prediction is returned as a JSON message.
4. If the service cannot be reached, the interface shows a connection error.

Prediction history is stored in the browser's local storage, which keeps the demo simple and avoids needing a database.

## Limitations

- **Very high R² needs context.** `Global_intensity` has a correlation of about 0.999 with the target, since power is largely determined by current and voltage. The scores above partly reflect that near-direct relationship rather than true forecasting ability.
- **Random split on time-series data.** Neighbouring minutes are highly similar, so a random split can overestimate performance. A chronological split would be a more realistic test.
- The data comes from a single household in France, so results may not generalise to other households.
- The frontend calls `http://127.0.0.1:5000`, so it is set up for local use only.

## Getting started

```bash
git clone https://github.com/<shahd-0>/household-energy-demand-predictor.git
cd household-energy-demand-predictor
pip install -r requirements.txt
```

1. Download the dataset into `data/` and run `notebooks/01_data_preprocessing.ipynb`.
2. Train the model: `cd backend && python train.py` (this creates `model.pkl`).
3. Start the service: `python app.py` (runs on `http://127.0.0.1:5000`).
4. Open `frontend/index.html` in your browser.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Returns predicted energy consumption (kWh) |
| POST | `/feedback` | Saves the actual value for a previous prediction |

Example request:

```json
POST /predict
{
  "hour": 18,
  "day_of_week": 2,
  "voltage": 240,
  "global_intensity": 8,
  "sub_metering_1": 1,
  "sub_metering_2": 2,
  "sub_metering_3": 10
}
```

Inputs are validated against allowed ranges; invalid input returns a `400` error.

## Future improvements

- Chronological train/test split and time-series cross-validation
- Retraining pipeline that uses the collected feedback data
- Docker container and cloud deployment
- Automated tests for the API

## Author

Shahd Islam, Computer Science (AI and Machine Learning), Al Akhawayn University



