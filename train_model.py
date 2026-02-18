import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)

crops = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"]
soils = ["Sandy", "Loamy", "Clay"]

crop_base = {
    "Wheat": 6,
    "Rice": 12,
    "Maize": 7,
    "Cotton": 8,
    "Sugarcane": 14
}

soil_factor = {
    "Sandy": 4,     # drains fast → more water
    "Loamy": 2,
    "Clay": -2      # retains water → less irrigation
}

data = []

for _ in range(200):

    crop = np.random.choice(crops)
    soil = np.random.choice(soils)

    temp = np.random.uniform(15, 45)
    humidity = np.random.uniform(30, 90)
    rainfall = np.random.uniform(0, 60)
    soil_moisture = np.random.uniform(10, 60)

    water = (
        crop_base[crop]
        + soil_factor[soil]
        + (temp * 0.25)
        - (humidity * 0.05)
        - (rainfall * 0.4)
        - (soil_moisture * 0.5)
    )

    water = max(0, water)

    data.append([crop, soil, temp, humidity, rainfall, soil_moisture, water])

df = pd.DataFrame(data, columns=[
    "Crop", "Soil", "Temperature", "Humidity",
    "Rainfall", "Soil_Moisture", "Water_Required"
])

X = df.drop("Water_Required", axis=1)
y = df["Water_Required"]

categorical = ["Crop", "Soil"]
numeric = ["Temperature", "Humidity", "Rainfall", "Soil_Moisture"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical),
    ("num", "passthrough", numeric)
])

model = RandomForestRegressor(n_estimators=150, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))

# Save model
joblib.dump(pipeline, "irrigation_model.pkl")

# ---------------- Feature Importance ----------------

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = pipeline.named_steps["model"].feature_importances_

plt.figure()
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")