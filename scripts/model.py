import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
class ModelTrainer:
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_xgboost(self):
        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(n_estimators=100, random_state=42))
        ])
        
        xgb_pipeline.fit(self.X_train, self.y_train)
        
        # Manually check if the XGBoost model is fitted
        if not is_model_fitted(xgb_pipeline.named_steps['model']):
            raise RuntimeError("The XGBoost model is not fitted.")
        
        # Predict on training data
        y_pred_train_xgb = xgb_pipeline.predict(self.X_train)

        # Predict on test data if available
        if self.X_test is not None:
            y_pred_test_xgb = xgb_pipeline.predict(self.X_test)
            return xgb_pipeline, y_pred_train_xgb, y_pred_test_xgb

        return xgb_pipeline, y_pred_train_xgb

    def train_random_forest(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(self.X_train, self.y_train)
        y_pred_train = pipeline.predict(self.X_train)
        
        if self.X_test is not None:
            y_pred_test = pipeline.predict(self.X_test)
            self.evaluate_model(self.y_test, y_pred_test)
        
        self.evaluate_model(self.y_train, y_pred_train)
        self.plot_feature_importances(pipeline)
        return pipeline

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")

    def plot_feature_importances(self, pipeline):
        rf_model = pipeline.named_steps['model']
        importances = rf_model.feature_importances_
        feature_names = self.X_train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)

        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Top Important Features from RandomForest')
        plt.gca().invert_yaxis()
        plt.show()

        y_pred_std = np.std([tree.predict(self.X_test) for tree in rf_model.estimators_], axis=0)
        confidence_interval = 1.96 * y_pred_std
        plt.figure(figsize=(10, 6))
        plt.errorbar(self.y_test.index, pipeline.predict(self.X_test), yerr=confidence_interval, fmt='o', ecolor='r', capthick=2, label="Confidence Interval")
        plt.scatter(self.y_test.index, self.y_test, color='blue', label='Actual Values', alpha=0.5)
        plt.title("Predictions with 95% Confidence Interval")
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.show()

    def save_model(self, pipeline, model_name):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_filename = f"./models/{model_name}_{timestamp}.pkl"
        os.makedirs('./models', exist_ok=True)
        joblib.dump(pipeline, model_filename)
        print(f"Model saved as {model_filename}")

    def create_lagged_data(self, data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    def train_lstm(self):
        sales = self.y_train.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_sales = scaler.fit_transform(sales.reshape(-1, 1))

        time_steps = 60
        X_lstm, y_lstm = self.create_lagged_data(scaled_sales, time_steps)
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

        X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
        X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(50, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), 
                            callbacks=[early_stop], verbose=2)

        y_pred_lstm = model.predict(X_test_lstm)
        y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
        y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm)

        mae_lstm = mean_absolute_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)
        mse_lstm = mean_squared_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)
        r2_lstm = r2_score(y_test_lstm_rescaled, y_pred_lstm_rescaled)

        print(f"LSTM Model - Test Set MAE: {mae_lstm:.2f}")
        print(f"LSTM Model - Test Set MSE: {mse_lstm:.2f}")
        print(f"LSTM Model - Test Set R²: {r2_lstm:.2f}")

        plt.figure(figsize=(14, 7))
        plt.plot(y_test_lstm_rescaled, label='Actual Sales', color='blue')
        plt.plot(y_pred_lstm_rescaled, label='Predicted Sales', color='orange')
        plt.title('Enhanced LSTM Model Predictions vs Actual Sales')
        plt.xlabel('Samples')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        lstm_model_filename = f"./models/lstm_model_{timestamp}.h5"
        model.save(lstm_model_filename)
        print(f'LSTM model serialized to {lstm_model_filename}')
