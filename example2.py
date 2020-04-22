from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
import numpy as np

data = [1 ,5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]

# series = Series(data)
# values = series.values

values = np.array(data)

values = values.reshape(-1, 1)

scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: {}, StandardDeviation: {}'.format(scaler.mean_, sqrt(scaler.var_)))

standarized = scaler.transform(values)
print(standarized)

inversed = scaler.inverse_transform(standarized)
print(inversed)