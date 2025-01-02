import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
import logging

warnings.filterwarnings("ignore")

class EDA:
    def __init__(self, train_path, test_path, store_path, log_path="../logs/eda.log"):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.store = pd.read_csv(store_path)
        logging.basicConfig(filename=log_path, level=logging.INFO)
        
    def load_data(self, file_path):
        return pd.read_csv(file_path)
    
    def data_overview(self, df):
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(df.info())
        print(df.describe())
    
    def check_missing_values(self, df):
        missing_values = df.isnull().sum()
        print(f"Missing Values:\n{missing_values}")
        msno.matrix(df)
        plt.show()
        
    def handle_missing_values(self):
        self.store['CompetitionDistance'].fillna(self.store['CompetitionDistance'].median(), inplace=True)
        self.store['CompetitionOpenSinceMonth'].fillna(self.store['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
        self.store['CompetitionOpenSinceYear'].fillna(self.store['CompetitionOpenSinceYear'].mode()[0], inplace=True)
        self.store['Promo2SinceWeek'].fillna(self.store['Promo2SinceWeek'].mode()[0], inplace=True)
        self.store['Promo2SinceYear'].fillna(self.store['Promo2SinceYear'].mode()[0], inplace=True)
        self.store['PromoInterval'].fillna('None', inplace=True)

        # Convert 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' into a single datetime column
        self.store['CompetitionOpenSince'] = pd.to_datetime(
            self.store['CompetitionOpenSinceYear'].astype(int).astype(str) + '-' + self.store['CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01',
            errors='coerce'
        )

        # Convert 'Promo2SinceYear' and 'Promo2SinceWeek' to a datetime format
        self.store['Promo2Since'] = pd.to_datetime(
            self.store['Promo2SinceYear'].astype(int).astype(str) + '-W' + self.store['Promo2SinceWeek'].astype(int).astype(str) + '-1',
            errors='coerce'
        )
   

    def plot_distribution(self, df, column, title, bins=50, color='blue', label=None, kde=True):
        """Plot the distribution of a column in the dataset."""
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=kde, bins=bins, color=color, label=label)
        plt.title(title)
        if label:
            plt.legend()
        plt.show()


    def visualize_outliers(self, df, columns, title):
        """Visualize outliers in the specified columns using boxplots."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[columns])
        plt.title(title)
        plt.show()


    def check_missing_values(self, df):
        """Check and visualize missing values in the dataset."""
        missing_values = df.isnull().sum()
        print(f"Missing Values:\n{missing_values}")
        msno.matrix(df)
        plt.show()
  
    def visualize_missing_data(self):
        self.check_missing_values(self.store)
        
    def distribution_plot(self, column, data, title, plot_type='countplot'):
        plt.figure(figsize=(8, 6))
        if plot_type == 'countplot':
            sns.countplot(x=column, data=data)
        elif plot_type == 'histplot':
            sns.histplot(data[column], kde=True)
        plt.title(title)
        plt.show()
        
    def holiday_sales_analysis(self):
        holiday_sales = self.train.groupby('StateHoliday')['Sales'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='StateHoliday', y='Sales', data=holiday_sales)
        plt.title('Sales During State Holidays')
        plt.show()
        
    def seasonal_sales_trends(self):
        self.train['Month'] = pd.to_datetime(self.train['Date']).dt.month
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Month', y='Sales', data=self.train)
        plt.title('Monthly Sales Distribution')
        plt.show()
        
    def correlation_analysis(self, df):
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation between features')
        plt.show()
        
    def promo_analysis(self):
        promo_sales = self.train.groupby('Promo')['Sales'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Promo', y='Sales', data=promo_sales)
        plt.title('Sales During Promotions vs No Promotions')
        plt.show()

    def assortment_sales_analysis(self):
        assortment_sales = self.train.merge(self.store, on='Store')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Assortment', y='Sales', data=assortment_sales)
        plt.title('Sales by Assortment Type')
        plt.show()
        
    def competition_effect_analysis(self):
        self.store['HasCompetition'] = self.store['CompetitionDistance'].apply(lambda x: 1 if x > 0 else 0)
        comp_effect = self.store.groupby('HasCompetition').mean()['CompetitionDistance']
        print(f"Effect of having competition on stores: \n{comp_effect}")

    def city_center_analysis(self):
        self.store['IsCityCenter'] = self.store['CompetitionDistance'].apply(lambda x: 1 if x <= 500 else 0)
        city_center_stores = self.store.groupby('IsCityCenter').mean()['CompetitionDistance']
        print(f"Effect of being in city center on stores: \n{city_center_stores}")

    def label_encode_columns(self):
        label_encoder = LabelEncoder()
        self.store['StoreType'] = label_encoder.fit_transform(self.store['StoreType'])
        self.store['Assortment'] = label_encoder.fit_transform(self.store['Assortment'])
        self.store['PromoInterval'].fillna('None', inplace=True)
        self.store['PromoInterval'] = label_encoder.fit_transform(self.store['PromoInterval'])

        correlation_matrix = self.store.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation between features")
        plt.show()
  

    def label_encode_columns(self):
        """Label encode categorical columns and visualize correlation matrix."""
        label_encoder = LabelEncoder()
        self.store['StoreType'] = label_encoder.fit_transform(self.store['StoreType'])
        self.store['Assortment'] = label_encoder.fit_transform(self.store['Assortment'])
        self.store['PromoInterval'].fillna('None', inplace=True)
        self.store['PromoInterval'] = label_encoder.fit_transform(self.store['PromoInterval'])

        correlation_matrix = self.store.corr()
        return correlation_matrix


    def promo2_analysis(self):
        """Analyze the effect of Promo2 on CompetitionDistance."""
        promo_data = self.store.groupby('Promo2').mean()['CompetitionDistance']
        print(f"Average CompetitionDistance with Promo2: \n{promo_data}")


    def promo2_distribution_analysis(self):
        """Visualize the distribution of Promo2."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Promo2', data=self.store)
        plt.title("Promo2 Distribution")
        plt.show()


    def data_overview(self, df):
        """Print a detailed overview of the dataset."""
        print(f"Dataset Shape: {df.shape}")
        print(f"Columns: {df.columns}")
        print(df.info())
        print(df.describe())


    def promo2_distribution_analysis(self):
        """Visualize the distribution of Promo2."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Promo2', data=self.store)
        plt.title("Promo2 Distribution")
        plt.show()

   

    def assortment_sales_analysis(self):
        """Analyze sales by assortment type."""
        assortment_sales = self.train.merge(self.store, on='Store')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Assortment', y='Sales', data=assortment_sales)
        plt.title('Sales by Assortment Type')
        plt.show()
    
    def complete_eda(self):
        self.handle_missing_values()
        self.visualize_missing_data()
        self.distribution_plot('StoreType', self.store, "Distribution of Stores by StoreType")
        self.distribution_plot('Assortment', self.store, "Distribution of Stores by Assortment")
        self.distribution_plot('CompetitionDistance', self.store, 'Distribution of CompetitionDistance', plot_type='histplot')
        self.holiday_sales_analysis()
        self.seasonal_sales_trends()
        self.correlation_analysis(self.train[['Sales', 'Customers']])
        self.promo_analysis()
        self.assortment_sales_analysis()
        self.competition_effect_analysis()
        self.city_center_analysis()
        self.label_encode_columns()
        logging.info("Exploratory data analysis completed for Task-1. Insights and visualizations generated.")
