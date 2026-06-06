import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
csv_file = BASE_DIR / "data" / "employee_salary.csv"
images_dir = BASE_DIR / "images"
results_file = BASE_DIR / "results.txt"

df = pd.read_csv(csv_file)

X = df[["Age", "Experience", "Education_Level"]]
y = df["Salary"]


def save_histogram(data, title, filename):
    data.hist(figsize=(8, 5))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(images_dir / filename)
    plt.close()


save_histogram(X, "Original Feature Distribution", "raw_features.png")

minmax_scaler = MinMaxScaler()
X_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(X),
    columns=X.columns
)

save_histogram(
    X_normalized,
    "Normalized Feature Distribution",
    "normalized_features.png"
)

standard_scaler = StandardScaler()
X_standardized = pd.DataFrame(
    standard_scaler.fit_transform(X),
    columns=X.columns
)

save_histogram(
    X_standardized,
    "Standardized Feature Distribution",
    "standardized_features.png"
)


linear_model = LinearRegression()

linear_model.fit(X, y)
linear_no_scaling_score = linear_model.score(X, y)

linear_model.fit(X_normalized, y)
linear_normalized_score = linear_model.score(X_normalized, y)

linear_model.fit(X_standardized, y)
linear_standardized_score = linear_model.score(X_standardized, y)


knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(X, y)
knn_no_scaling_score = knn_model.score(X, y)

knn_model.fit(X_normalized, y)
knn_normalized_score = knn_model.score(X_normalized, y)

knn_model.fit(X_standardized, y)
knn_standardized_score = knn_model.score(X_standardized, y)


results = f"""
Employee Salary Prediction - Feature Scaling Results

Linear Regression Scores:
No Scaling: {linear_no_scaling_score:.4f}
Normalization: {linear_normalized_score:.4f}
Standardization: {linear_standardized_score:.4f}

KNN Regression Scores:
No Scaling: {knn_no_scaling_score:.4f}
Normalization: {knn_normalized_score:.4f}
Standardization: {knn_standardized_score:.4f}

Observations:
1. Normalization scales values between 0 and 1.
2. Standardization transforms features around mean 0 and standard deviation 1.
3. Linear Regression may not show a big difference because it is less sensitive to feature scale.
4. KNN is more sensitive to feature scale because it uses distance between data points.
5. Feature scaling is very important for distance-based models and many deep learning models.
"""

results_file.write_text(results)

print(results)
print("Project completed successfully.")
