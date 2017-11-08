# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:30:57 2017

@author: Nguyen
"""

import tensorflow as tf


a = tf.add(3,5)

x = 2
y = 3

op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
useless = tf.multiply(x, op1)
op3 = tf.pow(op2,op1)

with tf.Session() as sess:
    op3, not_useless = sess.run([op3,useless])
    print(op3)
    print(not_useless)