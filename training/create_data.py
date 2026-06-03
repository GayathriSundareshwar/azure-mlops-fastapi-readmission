import pandas as pd
import numpy as np

np.random.seed(42)

num_samples = 1000

df = pd.DataFrame({
    "age": np.random.randint(20, 90, num_samples),
    "num_prior_admissions": np.random.randint(0, 10, num_samples),
    "length_of_stay": np.random.randint(1, 15, num_samples),
    "num_medications": np.random.randint(1, 20, num_samples),
    "has_chronic_condition": np.random.randint(0, 2, num_samples),
})

df["readmitted"] = (
    (df["num_prior_admissions"] > 4) |
    (df["length_of_stay"] > 7) |
    (df["has_chronic_condition"] == 1)
).astype(int)

df.to_csv(
    "data/readmission_data.csv",
    index=False
)

print("Saved data/readmission_data.csv")