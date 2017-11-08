# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:17:12 2017

@author: Nguyen
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

DATA_FILE = 'fire_theft.xls'

#Assemble graph


#Read in Data
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#Create placeholders for input X (number of fire) and label Y (number of theft)

X = tf.placeholder(data.dtype,name = 'inputs')
Y = tf.placeholder(data.dtype, name = 'labels')

weight = tf.Variable(0.0, dtype = data.dtype)
bias = tf.Variable(0.0, dtype = data.dtype)


Y_predicted = X * weight + bias

loss = tf.reduce_mean(tf.square(Y - Y_predicted, name='loss'))

#Optimizer
opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        total_loss = 0
        for x,y in data:
            _, l = sess.run([opt,loss], feed_dict={X:x, Y:y})
            total_loss += 1
        print("Epoch {0}: {1}".format(i, total_loss/n_samples))
        
    w_value, b_value = sess.run([weight,bias])
    
    
print(w_value, b_value)
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()