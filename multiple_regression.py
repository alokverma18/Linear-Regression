import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
#from sklearn.metrics import r2_score

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
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'])
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()


plt.scatter(cdf['CYLINDERS'], cdf['CO2EMISSIONS'])
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.show()

plt.scatter(cdf['FUELCONSUMPTION_COMB'], cdf['CO2EMISSIONS'])
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2 Emissions')
plt.show()


#Creating Train and Test Dataset
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
train_y = train[['CO2EMISSIONS']]

test_x = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
test_y = test[['CO2EMISSIONS']]


#Modelling
reg = linear_model.LinearRegression()
reg.fit(train_x, train_y)

print('\nCo-efficients:', reg.coef_)

#Model Evaluation
predictions = reg.predict(test_x)
print('\nVariance Score :', reg.score(test_x, test_y))
#print(r2_score(test_y, predictions))