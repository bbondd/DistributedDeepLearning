import numpy as np
import keras as k

model = k.Sequential()
model.add(k.layers.Conv2D(filters=3, kernel_size=2, input_shape=[28, 28]))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(units=1))
model.compile(optimizer='Adam', loss='mean_squared_error')

model.fit(x=np.zeros([3, 28, 28]), y=np.zeros([3, 1]))
