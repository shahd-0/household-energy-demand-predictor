import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"

FEATURES = [
    "hour",
    "day_of_week",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
TARGET = "energy consumption"


df = pd.read_csv(DATASET_PATH).dropna()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"R2: {r2_score(y_test, predictions):.4f}")

joblib.dump(model, MODEL_PATH)
