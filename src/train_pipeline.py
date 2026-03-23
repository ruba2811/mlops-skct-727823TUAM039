# Roll No: 727823TUAM039

import mlflow
import mlflow.sklearn
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import shutil

print("TRAINING STARTED")
print("Roll No: 727823TUAM039")
print("Timestamp:", datetime.now())


df = pd.read_csv("data/processed.csv")


X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


mlflow.set_experiment("SKCT_727823TUAM039_EnergyDataset")

best_r2 = -999


for i in range(12):

    with mlflow.start_run():

        print(f"\n--- Run {i+1} ---")

        
        mlflow.set_tag("student_name", "RUBA S")
        mlflow.set_tag("roll_number", "727823TUAM039")
        mlflow.set_tag("dataset", "EnergyDataset")

        start_time = time.time()

        
        if i % 2 == 0:
            model = LinearRegression()
            model_name = "LinearRegression"
        else:
            model = RandomForestRegressor(n_estimators=50)
            model_name = "RandomForest"

        
        model.fit(X_train, y_train)

        
        preds = model.predict(X_test)

        
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("r2_score", r2)

      
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        mlflow.log_param("model_name", model_name)

       
        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_name} | R2 Score: {r2}")

      
        if r2 > best_r2:
            best_r2 = r2

            
            if os.path.exists("models/best_model"):
                shutil.rmtree("models/best_model")

            mlflow.sklearn.save_model(model, "models/best_model")

print("\n========================")
print("BEST R2 SCORE:", best_r2)
print("Best model saved in models/best_model")