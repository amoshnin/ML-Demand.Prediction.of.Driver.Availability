import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import load, dump
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Driver Hours Prediction Pipeline')
parser.add_argument('--model_name', action='store', type=str, default = '../processed_data_and_models/xgboost_model.joblib' ,help='Prediction Model name')
parser.add_argument('--test_file_path', action='store', type=str, default = './../data/test.csv' ,help='Test File Path')
args = parser.parse_args()

model = load(args.model_name)

train = pd.read_csv('../processed_data_and_models/Training_Dataset.csv')
train.rename({'available_hours':'online_hours'}, axis = 1, inplace = True)
train = train[['date','driver_id','dayofweek','weekend','gender','age','number_of_kids','online_hours']]

train['dayofweek']= train['dayofweek'].astype('int64')
train['age']= train['age'].astype('int64')
train['number_of_kids']= train['number_of_kids'].astype('int64')
org_test = pd.read_csv(args.test_file_path)
test = org_test.sort_values(by = ['driver_id','date'])

try:
    test.drop(columns= ['online_hours'], inplace = True)
except:
    pass

driver_profile = pd.read_csv('../processed_data_and_models/driver.csv')
test = pd.merge(test, driver_profile, on=['driver_id'])

test['gender'].replace({'FEMALE':1, 'MALE':0}, inplace = True)
test['date'] = pd.to_datetime(test['date'])
test['dayofweek'] = test['date'].dt.dayofweek
test['weekend'] = test['dayofweek'].apply(lambda x: 0 if x < 5 else 1)

test.drop_duplicates(subset=['driver_id', 'date'], inplace = True)
driver_test_list = np.unique(test.driver_id.values)
train = train[train['driver_id'].isin(driver_test_list)]

test['online_hours'] = -1
test = test[train.columns]

test_data = pd.concat([train,test])

test_data['date'] = pd.to_datetime(test_data['date'])

test_data = test_data.set_index(
    ['date', 'driver_id']
).unstack().fillna(method = 'ffill').asfreq(
    'D'
).stack().sort_index(level=1).reset_index()

test_data['dayofweek'].fillna(test_data['date'].dt.dayofweek, inplace = True)
test_data['weekend'] = test_data['dayofweek'].apply(lambda x: 0 if x < 5 else 1)
test_data[test_data.driver_id==111556]
test_data = test_data.sort_values(by=['driver_id', 'date']).drop_duplicates(subset=['date','driver_id'])
test_data = test_data.set_index(['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age',
       'number_of_kids'])

test_data['lag_1'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(1)
test_data['lag_2'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(2)
test_data['lag_3'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(3)
test_data['lag_4'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(4)
test_data['lag_5'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(5)
test_data['lag_6'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(6)
test_data['lag_7'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(7)
test_data['rolling_mean'] = test_data.groupby(level=['driver_id'])['online_hours'].apply(lambda x: x.rolling(window = 7).mean()).shift(1)
test_data = test_data.reset_index(drop = False).dropna()
test_data = test_data[['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age',
       'number_of_kids', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
       'lag_5', 'lag_6', 'lag_7', 'rolling_mean','online_hours']]

def reset_test(test_data):
    test_data = test_data.set_index(['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age', 'number_of_kids'])
    test_data['lag_1'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(1)
    test_data['lag_2'] = test_data.groupby(level=['driver_id'])['lag_1'].shift(1)
    test_data['lag_3'] = test_data.groupby(level=['driver_id'])['lag_2'].shift(1)
    test_data['lag_4'] = test_data.groupby(level=['driver_id'])['lag_3'].shift(1)
    test_data['lag_5'] = test_data.groupby(level=['driver_id'])['lag_4'].shift(1)
    test_data['lag_6'] = test_data.groupby(level=['driver_id'])['lag_5'].shift(1)
    test_data['lag_7'] = test_data.groupby(level=['driver_id'])['lag_6'].shift(1)
    test_data = test_data.reset_index()
    test_data['rolling_mean'] = test_data[['lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7']].mean(axis=1)
    return test_data

start_date = datetime.strptime('2017-06-22','%Y-%m-%d')
end_date = pd.to_datetime(np.max(test_data['date'].values))
delta = timedelta(days=1)
while start_date <= end_date:
    chunk = test_data[test_data['date']==start_date]    
    X = chunk.iloc[:,2:-1]
    y = model.predict(X)
    y = [round(i,1) for i in y]
    chunk['online_hours'] = test_data.loc[test_data['date']==start_date, 'online_hours'] = y
    test_data = reset_test(test_data)
    start_date += delta

result = test_data[['date','driver_id', 'online_hours']]
result['date'] = pd.to_datetime(result['date'])
org_test['date'] = pd.to_datetime(org_test['date'])

pred_result = pd.merge(org_test[['date', 'driver_id']], result, on =['date','driver_id']).sort_values(by=['driver_id','date']).reset_index(drop = True)
print(pred_result.head(15))
pred_result.to_csv('../processed_data_and_models/PREDICTION.csv',index = False)
