"""
Suppose you have a polynomial equation y = 5x2 + 7x + 9. Now, you want to learn this
equation by a neural network. Write the possible solution in python and tensorflow. To be
more specific we have below input and target.
x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([5*i**2 + 7*i + 9 for i in x_train])
"""
from pyexpat import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import PIL

def polynomial(x):
    return 5*x**2 + 7*x + 9
#print(polynomial(5))

x_train = np.array([i for i in range(101)])
y_train = np.array([polynomial(i) for i in x_train])
#y_train = np.array([5*i**2 + 7*i + 9 for i in x_train])
print(x_train)
print(y_train)
plt.scatter(x_train, y_train)
plt.show()

x_test=x_train
y_test=y_train

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=200,input_shape=[1]))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(units=100))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(units=1))



model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train,epochs=500,batch_size=32,validation_data=( x_test,y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()

results=model.evaluate(x_train,y_train)
print(results)



