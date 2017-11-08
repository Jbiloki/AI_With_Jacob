# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:19:14 2017

@author: Nguyen
"""

import tensorflow as tf

a = tf.Variable(2, name = 'scalar')

b = tf.Variable([2,3], name = 'vector')

c = tf.Variable([[0,1],[2,3]], name = 'matrix')

W = tf.Variable(tf.zeros([784,10]))

#Initialize variables all at once

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
#Or init subset

init_ab = tf.variables_initializer([a,b], name = 'init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
    
    
#Or single variable
#W = tf.Variable(tf.truncated_normal([700,10]))
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    #sess.run(W.initializer)
    sess.run(assign_op) #This will initialize our variable
    print(W.eval())
    
    
#Create a variable whose original value is 2

my_var = tf.Variable(2, name='my_var')

#assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_times_two)
    print(my_var.eval())
    print(my_var_times_two.eval())
    
    
    
my_var = tf.Variable(10)

'''
Each session has it's own variable if we run this in two different sessions 
they will both be different operations on the origional value of 10
'''
with tf.Session() as sess:
    sess.run(my_var.initializer)
    #Increment by 10
    sess.run(my_var.assign_add(10)) # 20
    # decrement by 2
    sess.run(my_var.assign_sub(2)) #18
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    