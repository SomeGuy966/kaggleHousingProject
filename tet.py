from pyexpat import features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

home_data_path = 'home-data-for-ml-course/train.csv'
home_data = pd.read_csv(home_data_path)

# Including the features that I want in my data
features = ['LotFrontage',
            'LotArea',
            'Street',
            'LotShape',
            'LandContour',
            'Utilities',
            'Neighborhood',
            'YearBuilt',
            'RoofStyle',
            'RoofMatl',
            'Exterior1st',
            'Exterior2nd',
            'MasVnrType',
            'MasVnrArea',
            'ExterQual',
            'ExterCond',
            'Foundation',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinSF2',
            'TotalBsmtSF',
            'Heating',
            'HeatingQC',
            'CentralAir',
            '1stFlrSF',
            '2ndFlrSF',]

# Gathering data from home_data
X = home_data[features]
y = home_data['SalePrice']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor(random_state=0)
model.fit(train_X, train_y)
forecasted_y = model.predict(val_X)
mae = mean_absolute_error(val_y, forecasted_y)

print(mae)

