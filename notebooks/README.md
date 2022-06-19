# Driver Online_Hours Forecast
+ Language Used: Python | Version >=3.5
+ IPython Notebooks: 4
+ HTML Files: Are data analysis for Datasets. 

1. Data Analysis: To analyse driver, pings and test data

##### DriversProfile.csv
+ No NULL or Missing Values
+ Duplicate Values subset=[driver_id], count: 3
+ 2497: Unique Driver Ids
+ Driver Age varies from 18 to 75 | Median at 31years
+ Gender column is categorical with two classes Male and Female; Male= 1894, Female= 606
+ Age and Number of Children has +ve correlation

##### Pings_Data
+ 2 Columns | Driverid and pingtimestamp
+ Data Contains Duplicate Values: 79086 count
+ No NaN values found in Data.
+ 2480: Unique Driver Ids
+ No considering: Removing 22nd June Data

##### Test Data
+ Contains duplicate subset = ['driver_id', 'date'] 
+ Unique rows = 17497


Here Pings_Data is combined on basis of pings received every 15secs by driver's mobile indicating that he is available. I am assuming a PING_THRESHOLD as 60secs considering system faults for missing pings. Driver cannot be missing for less than 60secs. 

2. Preparing Training Data
+ Joined Driver_Profile and Driver Combined Ping Data
+ Including time features like dayofweek, weekend

3. Model Training
> Models used: Using Linear Regression, Decision Tree, Random Forest, Xgboost Regressor
+ Iteration1: ['dayofweek','weekend','gender','age','number_of_kids']
+ Iteration2: Lag Features Added
+ Iteration3: Rolling Window
+ Iteration4: Both Lag and Rolling Window considered

> BEST Model: XGBOOST with Features: ['dayofweek', 'weekend', 'gender', 'age',
       'number_of_kids', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
       'lag_5', 'lag_6', 'lag_7', 'rolling_mean']
> Train RMSE: 1.65; Test RMSE: 2.01

4. TestScript

### AUTHOR:
Shaurya Uppal [shauryauppal00111@gmail.com]