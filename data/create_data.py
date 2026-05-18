import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

n = 1000

# synthetic features
X1 = np.random.rand(n) * 10
X2 = np.random.rand(n) * 5
X3 = np.random.rand(n) * 20

# synthetic target (linear + noise)
y = 3*X1 + 2*X2 + 0.5*X3 + np.random.randn(n) * 2

df = pd.DataFrame({
    "feature1": X1,
    "feature2": X2,
    "feature3": X3,
    "target": y
})

df.to_csv("data/housing_v1.csv", index=False)

print("Dataset created at data/housing_v1.csv")
