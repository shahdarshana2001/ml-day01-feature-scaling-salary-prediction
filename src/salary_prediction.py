import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
csv_file = BASE_DIR / "data" / "employee_salary.csv"
images_dir = BASE_DIR / "images"

df = pd.read_csv(csv_file)

print("Dataset Loaded Successfully")
print(df.head())

print("\nSummary Statistics")
print(df.describe())

df[["Age", "Experience", "Education_Level"]].hist(figsize=(8, 5))

plt.tight_layout()
plt.savefig(images_dir / "raw_features.png")
plt.show()
print("raw_features.png saved inside images folder")