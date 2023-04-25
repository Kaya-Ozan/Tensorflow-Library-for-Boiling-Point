import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from bokeh.plotting import figure, show
from bokeh.io import output_file

data = pd.DataFrame({'Moleküler Ağırlık': [18.015, 44.01, 16.042, 28.97, 2.016, 32.00],
                     'Yoğunluk': [0.998, 1.977, 0.717, 1.429, 0.0899, 1.429],
                     'Kaynama Noktası (K)': [373.15, 216.58, 111.66, 59.15, 20.28, 161.49]})

X = data[['Moleküler Ağırlık', 'Yoğunluk']].values
y = data['Kaynama Noktası (K)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(2,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=500, batch_size=4,verbose=0)

y_pred = model.predict(X_test_scaled)

output_file("ozan.html")
p = figure(title="Gerçek Kaynama Noktası ve Tahminleri",
           x_axis_label="Gerçek Kaynama Noktası (K)",
           y_axis_label="Tahmin Edilen Kaynama Noktası (K)",
           width=600, height=400)

p.circle(y_test, y_pred[:, 0], size=8, color="blue", legend_label="Tahminler")
p.line([0, 400], [0, 400], color="red", legend_label="Ideal Çizgi")

p.legend.location = "top_left"
show(p)