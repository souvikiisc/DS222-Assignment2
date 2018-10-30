from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
# import h5py
import pickle
from scipy import sparse
parameter_servers = ["10.24.1.218:2225",
		    "10.24.1.219:2225"]
workers = [ "10.24.1.220:2225",
      "10.24.1.221:2225",
    "10.24.1.222:2225"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS


server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

batch_size = 4096
learning_rate = 0.002
training_epochs = 100
logs_path = "/home/souvikk/project/assignments/assignment2/stale_10"

if FLAGS.job_name == "ps":
    server.join()
    print("ps initialized !!")
elif FLAGS.job_name == "worker":
    print("data load")
    y_train = np.load("train_labels.npy").astype(np.float32)
    y_test = np.load("test_labels.npy").astype(np.float32)
    training_set = sparse.load_npz('train_set.npz')
    test_set = sparse.load_npz('test_set.npz')
    print("done")

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):

        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = 0.002
        training_epochs = 100
        batch_size = 2048
        display_step = 1

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, training_set.shape[1]])
        y = tf.placeholder(tf.float32, [None, 50])
        # Set model weights
        W = tf.Variable(tf.random_normal([training_set.shape[1], 50]))
        b = tf.Variable(tf.random_normal([50]))

        weight = tf.reshape(tf.reduce_sum(y, 0) / tf.reduce_sum(tf.reduce_sum(y, 0)), [1, 50])

        # Construct model
        pred = (tf.matmul(x, W) + b)  # Softmax
        weight_per_label = tf.transpose(tf.matmul(y, tf.transpose(weight)))

        xent = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
        loss = tf.reduce_mean(xent)  # shape 1
        regularizer = tf.nn.l2_loss(W)
        cost = tf.reduce_mean(loss + 0.01 * regularizer)
        # Gradient Descent
        grad_op = tf.train.AdamOptimizer(learning_rate)
        rep_op = tf.contrib.opt.DropStaleGradientOptimizer(grad_op,
                                                         staleness=15,use_locking=True)
        optimizer = rep_op.minimize(cost, global_step=global_step)

        prediction = tf.nn.softmax(tf.matmul(x, W) + b)
        # test_prediction = tf.nn.softmax(tf.matmul(test_set, W) + b)


        def accuracy(predictions, labels):
            p = 0
            k = predictions.shape[0]
            for i in range(predictions.shape[0]):
                if (labels[i, np.argmax(predictions[i, :])] != 0):
                    p = p + 1
            return p


        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        sess = tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                                 hooks=hooks)
        print('Starting training on worker %d' % FLAGS.task_index)
        # while not sess.should_stop():

        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        epoch_cost = {}
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(training_set.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = training_set[i * batch_size:i * batch_size + batch_size, :].toarray()
                batch_ys = y_train[i * batch_size:i * batch_size + batch_size, :]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            cost_x = epoch_cost.get(FLAGS.task_index, np.zeros((training_epochs, 1)))
            cost_x[epoch, 0] = avg_cost
            epoch_cost[FLAGS.task_index] = cost_x

        print("Optimization Finished!")
        print("Total Time: %3.2fs" % float(time.time() - start_time))
        acc = 0.0
        pred = sess.run(prediction, feed_dict={x: training_set.toarray()})
        acc += accuracy(pred, y_train)
        print("train_accuracy=", acc / y_train.shape[0])
        # begin_time = time.time()
        # acc = accuracy(sess.run(prediction, feed_dict={x: test_set}),y_test)
        # end_time = time.time()
        acc = 0.0
        begin_time = time.time()
        pred = sess.run(prediction, feed_dict={x: test_set.toarray()})
        acc += accuracy(pred, y_test)
        end_time = time.time()
        print("test_accuracy=", acc / y_test.shape[0])
        print("test time: %3.2fs" % float(end_time - begin_time))
        file_name = 'st_cost_epochs_' + str(FLAGS.task_index)
        f = open(file_name, 'w')
        pickle.dump(epoch_cost, f, protocol=pickle.HIGHEST_PROTOCOL)

        sess.close()
        print('Session from worker %d closed cleanly' % FLAGS.task_index)

