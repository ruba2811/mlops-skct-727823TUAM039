# Roll No: 737823TUAM039

import pandas as pd
from datetime import datetime
import os
print("Current working directory:", os.getcwd())

print("Roll No: 727823TUAM039")
print("Timestamp:", datetime.now())

df = pd.read_csv("data/energy.csv")


df.columns = ["datetime", "price"]


df["datetime"] = pd.to_datetime(df["datetime"])

# Update 1
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month


df = df.drop("datetime", axis=1)

df.to_csv("data/processed.csv", index=False)
#update 2
print("Data preprocessing done")
