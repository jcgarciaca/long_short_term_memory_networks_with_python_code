from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np

'''
data = [x for x in range(10, 110, 10)]
series = Series(data)
print(series)

values = series.values
'''
values = np.linspace(10, 100, 10)
print(type(values), values.shape, values)

values = values.reshape(len(values), 1)
print(values.shape, values)

scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(values)
print(type(scaler), scaler)
print('Min: {}, Max: {}'.format(scaler.data_min_, scaler.data_max_))

normalized = scaler.transform(values)
print(normalized)

print(scaler.transform(np.array([50]).reshape(-1, 1)))

inversed = scaler.inverse_transform(normalized)
print(inversed)