# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:36:05 2017

@author: Nguyen
"""

import tensorflow as tf


#Create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5,5,5], tf.float32)

#use the placeholder as you would a constant or a variable
c = a + b # short for tf.add(a,b)

with tf.Session() as sess:
    # feed [1,2,3] into placeholder a via the dict {a:[1,2,3]}
    # fetch value of c
    print (sess.run(c ,{a: [1,2,3]}))


'''Feeding values to TF ops'''

# create operations, tensors, etc (using the default graph)

a = tf.add(2,5)
b = tf.multiply(a,3)

with tf.Session() as sess:
    #define a dictionary that says to replace the value of 'a' with 15
    replace_dict = {a:15}
    
    #Run session, passing in 'replaec_dict' as the valeu to 'feed_dict'
    
    print(sess.run(b))#,feed_dict = replace_dict))
    #print(tf.get_default_graph().as_graph_def()) #Print graph variables in json form