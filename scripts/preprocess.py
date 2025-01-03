# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path

    def load_data(self):
        self.train = pd.read_csv(self.train_path, low_memory=False)
        self.test = pd.read_csv(self.test_path, low_memory=False)
        self.store = pd.read_csv(self.store_path, low_memory=False)
        return self.train, self.test, self.store

    def summarize_data(self):
        print(self.train.dtypes)
        print(self.train.describe())
        print(self.test.dtypes)
        print(self.store.dtypes)
        print(self.store.describe())
        print(self.train.isna().sum())
        print(self.test.isna().sum())
        print(self.store.isnull().sum())

    def merge_datasets(self):
        train_merged = pd.merge(self.train, self.store, on='Store', how='left')
        test_merged = pd.merge(self.test, self.store, on='Store', how='left')
        self.df = pd.concat([train_merged, test_merged], axis=0)
        return self.df

    def handle_missing_values(self):
        self.df['Sales'].fillna(0, inplace=True)
        self.df['Customers'].fillna(0, inplace=True)
        self.df['Open'].fillna(0, inplace=True)
        self.df['CompetitionDistance'].fillna(self.df['CompetitionDistance'].median(), inplace=True)
        self.df['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
        self.df['CompetitionOpenSinceYear'].fillna(2000, inplace=True)
        self.df['Promo2SinceWeek'].fillna(0, inplace=True)
        self.df['Promo2SinceYear'].fillna(0, inplace=True)
        self.df['PromoInterval'].fillna('No Promo', inplace=True)
        if 'Id' in self.df.columns:
            self.df.drop(columns=['Id'], inplace=True)
        return self.df

    def feature_engineering(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Day'] = self.df['Date'].dt.day
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        self.df['IsBeginningOfMonth'] = self.df['Day'].apply(lambda x: 1 if x <= 7 else 0)
        self.df['IsMidMonth'] = self.df['Day'].apply(lambda x: 1 if 8 <= x <= 21 else 0)
        self.df['IsEndOfMonth'] = self.df['Day'].apply(lambda x: 1 if x > 21 else 0)

        categorical_features = ['StoreType', 'Assortment', 'PromoInterval']
        label_encoder = LabelEncoder()
        for col in categorical_features:
            self.df[col] = label_encoder.fit_transform(self.df[col].astype(str))

        scaled_columns = ['CompetitionDistance', 'Day', 'WeekOfYear', 'Month', 'Year']
        scaler = StandardScaler()
        self.df[scaled_columns] = scaler.fit_transform(self.df[scaled_columns])

        self.X = self.df.drop(columns=['Sales', 'Date'])
        self.y = self.df['Sales']
        non_numeric_columns = self.X.select_dtypes(include=['object']).columns
        self.X_encoded = pd.get_dummies(self.X, columns=non_numeric_columns)
        return self.X_encoded, self.y
