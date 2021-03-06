

from __future__ import print_function

import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import sys
import time
import h5py

y_train=np.load("train_l.npy").astype(np.float32)
y_test=np.load("test_l.npy").astype(np.float32)
h5f1 = h5py.File('training_set.h5','r')
x_train = h5f1['d1'][:]
h5f2 = h5py.File('test_set.h5','r')
x_test = h5f2['d2'][:]
print("data_loaded")



print (x_train.shape)
print (y_train.shape)


# Parameters
starter_learning_rate = 0.001
training_epochs = 150
batch_size = 4096
display_step = 1

global_step = tf.Variable(0, trainable=False)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, x_train.shape[1]]) 
y = tf.placeholder(tf.float32, [None, 50]) 
# Set model weights
W = tf.Variable(tf.random_normal([x_train.shape[1], 50]))
b = tf.Variable(tf.random_normal([50]))

weight=tf.reshape(tf.reduce_sum(y,0)/tf.reduce_sum(tf.reduce_sum(y,0)),[1,50])

# Construct model
pred = (tf.matmul(x, W) + b) # Softmax
weight_per_label = tf.transpose( tf.matmul(y
                           , tf.transpose(weight)) ) 

xent = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
loss = tf.reduce_mean(xent) #shape 1
regularizer = tf.nn.l2_loss(W)
cost = tf.reduce_mean(loss + 0.01 * regularizer)
# Gradient Descent
total_batch = int(x_train.shape[0]/batch_size)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           total_batch, 1.04, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)

#test_prediction = tf.nn.softmax(tf.matmul(x_test, W) + b)
#train_prediction = tf.nn.softmax(tf.matmul(x_train, W) + b)
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
def accuracy(predictions, labels):
    p=0
    k=predictions.shape[0]
    for i in range(predictions.shape[0]):
        if (labels[i,np.argmax(predictions[i,:])]!=0):
           p=p+1
           #print("correct")
        #else:
           #print("wrong")
    #print (np.argmax(predictions[50,:]),predictions[50,:],labels[50,:]) 
    return (100.0 * p/ labels.shape[0])
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
save_cost=np.zeros((training_epochs,1))
lr=np.zeros((training_epochs,1))
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    begin_time = time.time()
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs=x_train[i*batch_size:i*batch_size+batch_size,:]
	    batch_ys = y_train[i*batch_size:i*batch_size+batch_size,:]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        #print (accuracy(test_prediction.eval(),y_test))
        print (accuracy(sess.run(prediction, feed_dict={x: x_test}),y_test))
        print (learning_rate.eval())
        save_cost[epoch,0]=avg_cost
        lr[epoch,0]=learning_rate.eval()


    print("Optimization Finished!")
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    print ("train_accuracy=",accuracy(sess.run(prediction, feed_dict={x: x_train}),y_train))
    print ("test_accuracy=",accuracy(sess.run(prediction, feed_dict={x: x_test}),y_test))
    np.save("cost_inc",save_cost)
    






