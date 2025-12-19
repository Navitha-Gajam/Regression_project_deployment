'''
In this file we are going to read data and make connections  based on requirement
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')
import logging
from logfile import setup_logging
logger = setup_logging('main')

class DataLoader:
    def __init__(self, exercise_path, calories_path,logger):
        self.exercise_path = exercise_path
        self.calories_path = calories_path
        self.logger = logger

    def load_data(self):
        self.logger.info("Loading exercise and calories datasets")
        exercise = pd.read_csv(self.exercise_path)
        calories = pd.read_csv(self.calories_path)

        # 🔹 Log shapes
        self.logger.info(f"Exercise dataset shape: {exercise.shape}")
        self.logger.info(f"Calories dataset shape: {calories.shape}")

        self.logger.info("Datasets loaded successfully")
        return exercise, calories


class DataPreprocessor:
    def __init__(self, exercise, calories, logger):
        self.exercise = exercise
        self.calories = calories
        self.logger = logger

    def preprocess(self):
        self.logger.info("Merging datasets on User_ID")

        df = pd.merge(self.exercise, self.calories, on="User_ID")

        # 🔹 Log merged shape
        self.logger.info(f"Merged dataset shape: {df.shape}")

        self.logger.info("Dropping unnecessary columns")
        df = df.reset_index()
        df = df.drop(["index", "User_ID"], axis=1)

        self.logger.info("Encoding Gender")
        gender = pd.get_dummies(df["Gender"], drop_first=True)
        df = df.drop("Gender", axis=1)
        df = pd.concat([df, gender], axis=1)

        self.logger.info(f"Final dataset shape after preprocessing: {df.shape}")

        return df


class FeatureEngineer:
    def __init__(self, df, logger):
        self.df = df
        self.logger = logger
        self.scaler = StandardScaler()

    def split_and_scale(self):
        X = self.df.drop("Calories", axis=1)
        y = self.df["Calories"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        self.logger.info(f"X_train: {X_train.shape}")
        self.logger.info(f"X_test: {X_test.shape}")
        self.logger.info(f"y_train: {y_train.shape}")
        self.logger.info(f"y_test: {y_test.shape}")

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

class ModelTrainer:
    def __init__(self, logger):
        self.model = LinearRegression()
        self.logger = logger

    def train(self, X_train, y_train):
        self.logger.info("Training Linear Regression model")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, X_test, y_train, y_test):
        train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = self.model.score(X_test, y_test)

        self.logger.info(f"Train R2 Score: {train_r2}")
        self.logger.info(f"Test R2 Score: {test_r2}")

    # ✅ SAVE MODEL
    def save_model(self, file_path="calories.pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Model saved to {file_path}")

    # ✅ LOAD MODEL
    def load_model(self, file_path="calories.pkl"):
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.logger.info(f"Model loaded from {file_path}")

    # ✅ PREDICT
    def predict(self, X):
        return self.model.predict(X)





if __name__ == "__main__":

    logger = setup_logging("Calories_Burnt_Project")
    logger.info("Project started")
    # Load data
    loader = DataLoader("exercise.csv", "calories.csv", logger)
    exercise, calories = loader.load_data()

    #preprocessing data
    processor = DataPreprocessor(exercise, calories, logger)
    df = processor.preprocess()

    #spliting data for training and testing
    fe = FeatureEngineer(df, logger)
    X_train, X_test, y_train, y_test = fe.split_and_scale()

    trainer = ModelTrainer(logger)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_train, X_test, y_train, y_test)

    trainer.save_model("calories.pkl")

    # Load model
    trainer.load_model("calories.pkl")

    # Predict new data (already scaled!)
    prediction = trainer.predict([[1, 2, 3, 4, 5, 6, 7]])[0]
    logger.info(f"Predicted Calories: {prediction}")