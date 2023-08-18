import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

path = 'FuelConsumptionCo2.csv'
df = pd.read_csv(path)

print('\nReading the Data :')
print(df.head())


print('\nData Exploration :')
print(df.describe())


print('\nFeatures Selection :')
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head())


#Data Plotting
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2 Emissions')
plt.show()


#Creating Train and Test Dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = train[['FUELCONSUMPTION_COMB']]
train_y = train[['CO2EMISSIONS']]

test_x = test[['FUELCONSUMPTION_COMB']]
test_y = test[['CO2EMISSIONS']]


#Modelling

print('\nModelling using FUELCONSUMPTION_COMB :')
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)

print('Coefficient : ', reg.coef_[0][0])
print('Intercept : ', reg.intercept_[0])

#Output Plotting
plt.scatter(train_x, train_y)
plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')

predictions = reg.predict(test_x)

print('\nModel Evaluation :')
print('R2-Score : ', r2_score(test_y, predictions))

