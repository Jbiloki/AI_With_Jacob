# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:51:23 2017

@author: Nguyen
"""

import tensorflow as tf

a = tf.constant([2,2], name='a')

b = tf.constant([[0,1],[2,3]], name='b')

x = tf.add(a,b, name='add')

y = tf.multiply(a,b, name='multiply')

with tf.Session() as sess:
    x,y = sess.run([x,y])
    #This runs our tensorboard
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(x,y)
#writer.close() #Must close writer after use