import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset
data = pd.DataFrame({
    'Store': [1, 2, 3, 4, 5],
    'DayOfWeek': [5, 3, 4, 6, 2],
    'Customers': [550, 620, 400, 720, 560],
    'Open': [1, 1, 0, 1, 1],
    'Promo': [1, 0, 1, 1, 0],
    'StateHoliday': [0, 0, 0, 0, 0],
    'SchoolHoliday': [0, 1, 1, 0, 0],
    'StoreType': [1, 2, 1, 3, 2],
    'Assortment': [1, 2, 3, 1, 2],
    'CompetitionDistance': [500.0, 1000.0, 1500.0, 2000.0, 1200.0],
    'CompetitionOpenSinceMonth': [9, 12, 5, 3, 7],
    'CompetitionOpenSinceYear': [2012, 2010, 2015, 2014, 2011],
    'Promo2': [0, 1, 1, 0, 1],
    'Promo2SinceWeek': [45, 13, 14, 30, 20],
    'Promo2SinceYear': [2013, 2014, 2013, 2015, 2012],
    'PromoInterval': [2, 3, 0, 1, 4],
    'Day': [15, 16, 14, 18, 19],
    'WeekOfYear': [32, 45, 50, 12, 22],
    'Month': [5, 8, 7, 3, 4],
    'Year': [2016, 2015, 2017, 2018, 2014],
    'IsWeekend': [0, 1, 0, 0, 1],
    'IsBeginningOfMonth': [0, 1, 0, 0, 1],
    'IsMidMonth': [1, 0, 0, 1, 0],
    'IsEndOfMonth': [0, 0, 1, 0, 0],
    'Sales': [1000, 1500, 1200, 1700, 1300]  # Target column
})

# Define feature columns (X) and target column (y)
X = data[['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday',
          'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance',
          'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
          'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Day',
          'WeekOfYear', 'Month', 'Year', 'IsWeekend', 'IsBeginningOfMonth',
          'IsMidMonth', 'IsEndOfMonth']]

y = data['Sales']

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)