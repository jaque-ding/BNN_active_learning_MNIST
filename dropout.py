from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import range
import numpy as np
import scipy as sp
import random
import math as m # for pi
import time

# random.seed(5437)

random.seed(5820)
import scipy.io

# Bayes by Backprop or Dropout
bbb = 0

is_saving=1

gpu=0

is_weight_decay = 1

toy_mode = 0

val_mode = 0
random_acquisitions = 1


acquisition_iterations = 98#100 #98

# note: use a large number of dropout iterations
dropout_iterations = 100 #100 #100

test_iterations = 50

Experiments = 3

# Parameters
learning_rate = 0.05
num_epochs = 100 # note: use large number of epochs
batch_size = 120
display_step = 10
weight_decay_constant = 3.5 # will be divided by train_size


# Network Parameters
n_hidden_1 = 400 # 1st layer number of neurons
n_hidden_2 = n_hidden_1 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout_rate = 0.5


score = 0
all_accuracy = 0

Queries = 10

if toy_mode:

    Experiments = 2
    stochastic_iterations = 2
    test_iterations = 2
    acquisition_iterations = 2
    num_epochs = 2
    is_saving = 0

accuracy_experiments_sum = np.zeros(shape=(acquisition_iterations + 1))

time_start = time.time()

for e in range(Experiments):

    print('Experiment Number ', (e+1))

    def one_hot(indices):
        new = np.zeros((indices.size, indices.max()+1))
        new[np.arange(indices.size),indices] = 1
        return new
    # the data, shuffled and split between train and test sets
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train_all = np.concatenate((mnist.train.images, mnist.validation.images))
    x_test = mnist.test.images
    y_train_all = np.concatenate((mnist.train.labels, mnist.validation.labels))
    y_test = mnist.test.labels


    # reshape images from one long vector to square matrices
    # x_train_all = x_train_all.reshape(x_train_all.shape[0], 1, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    # reshape images from one long vector to still one long vector
    x_train_all = x_train_all.reshape(x_train_all.shape[0], num_input)
    x_test = x_test.reshape(x_test.shape[0], num_input)

    # shuffle indices
    random_split = np.asarray(random.sample(range(0, x_train_all.shape[0]), x_train_all.shape[0]))

    # shuffle train set
    x_train_all = x_train_all[random_split, :]
    y_train_all = y_train_all[random_split]

    # 5.000 are the valation set
    x_val = x_train_all[10000:15000, :]
    y_val = y_train_all[10000:15000]

    # 40.000 are the pool set
    x_pool = x_train_all[20000:600000, :]
    y_pool = y_train_all[20000:600000]

    # 10.000 are where initial 20 are taken from
    x_train_all = x_train_all[:10000, :]
    y_train_all = y_train_all[0:10000]


    # try to make initial training data have an equal distribution of classes
    # get indices for each digit, then get only first 2 for each digit to make up
    # the initial train set of 20 samples
    idx_0 = np.array(np.where(y_train_all == 0)).T
    idx_0 = idx_0[0:2, 0]
    x_0 = x_train_all[idx_0, :]
    y_0 = y_train_all[idx_0]
    idx_1 = np.array(np.where(y_train_all == 1)).T
    idx_1 = idx_1[0:2, 0]
    x_1 = x_train_all[idx_1, :]
    y_1 = y_train_all[idx_1]
    idx_2 = np.array(np.where(y_train_all == 2)).T
    idx_2 = idx_2[0:2, 0]
    x_2 = x_train_all[idx_2, :]
    y_2 = y_train_all[idx_2]
    idx_3 = np.array(np.where(y_train_all == 3)).T
    idx_3 = idx_3[0:2, 0]
    x_3 = x_train_all[idx_3, :]
    y_3 = y_train_all[idx_3]
    idx_4 = np.array(np.where(y_train_all == 4)).T
    idx_4 = idx_4[0:2, 0]
    x_4 = x_train_all[idx_4, :]
    y_4 = y_train_all[idx_4]
    idx_5 = np.array(np.where(y_train_all == 5)).T
    idx_5 = idx_5[0:2, 0]
    x_5 = x_train_all[idx_5, :]
    y_5 = y_train_all[idx_5]
    idx_6 = np.array(np.where(y_train_all == 6)).T
    idx_6 = idx_6[0:2, 0]
    x_6 = x_train_all[idx_6, :]
    y_6 = y_train_all[idx_6]
    idx_7 = np.array(np.where(y_train_all == 7)).T
    idx_7 = idx_7[0:2, 0]
    x_7 = x_train_all[idx_7, :]
    y_7 = y_train_all[idx_7]
    idx_8 = np.array(np.where(y_train_all == 8)).T
    idx_8 = idx_8[0:2, 0]
    x_8 = x_train_all[idx_8, :]
    y_8 = y_train_all[idx_8]
    idx_9 = np.array(np.where(y_train_all == 9)).T
    idx_9 = idx_9[0:2, 0]
    x_9 = x_train_all[idx_9, :]
    y_9 = y_train_all[idx_9]

    x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)

    x_train_initial_indices = np.concatenate((idx_0, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7, idx_8, idx_9))
    print('indices of initial training set:', x_train_initial_indices)

    # make data into float 32s, divide pixel values by 255, and labels into categorical variables

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_pool = x_pool.astype('float32')
    x_test = x_test.astype('float32')

    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_pool = one_hot(y_pool)
    y_test = one_hot(y_test)

    # loss values for each epoch
    pool_train_loss = np.zeros(shape=(num_epochs, 1))
    pool_val_loss = np.zeros(shape=(num_epochs, 1))
    pool_train_acc = np.zeros(shape=(num_epochs, 1))
    pool_val_acc = np.zeros(shape=(num_epochs, 1))
    x_pool_all = np.zeros(shape=(1))

    print('training Model Without Acquisitions in Experiment ', (e+1) )


    def next_batch(data, labels, num):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


    def get_accuracy(out, true_one_hot):
        predictions = np.argmax(out, axis=1)
        true = np.argmax(true_one_hot, axis = 1)
        accuracy =  np.mean(predictions == true)
        return accuracy



    # Reset graph, recreate placeholders and dataset.
    tf.reset_default_graph()
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    if gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):

        # tf Graph input
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)

        # Store layers weight & bias
        # w1 = tf.Variable(tf.random_normal(shape=[num_input, n_hidden_1]))
        # w2 = tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2]))
        # w3 = tf.Variable(tf.random_normal(shape=[n_hidden_2, num_classes]))


        # w1 = tf.get_variable(
        #     'w1', shape=[num_input, n_hidden_1], dtype=tf.float32,
        #     initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(num_input)))
        # w2 = tf.get_variable(
        #     'w2', shape=[n_hidden_1, n_hidden_2], dtype=tf.float32,
        #     initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(n_hidden_1)))
        # w3 = tf.get_variable(
        #     'mw3', shape=[n_hidden_2, num_classes], dtype=tf.float32,
        #     initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(n_hidden_2)))


        # w1 = tf.get_variable(
        #     'w1', shape=[num_input, n_hidden_1], dtype=tf.float32,
        #     initializer=tf.glorot_normal_initializer())
        # w2 = tf.get_variable(
        #     'w2', shape=[n_hidden_1, n_hidden_2], dtype=tf.float32,
        #     initializer=tf.glorot_normal_initializer())
        # w3 = tf.get_variable(
        #     'mw3', shape=[n_hidden_2, num_classes], dtype=tf.float32,
        #     initializer=tf.glorot_normal_initializer())

    #     w1 = tf.Variable(tf.random_normal(shape=[num_input, n_hidden_1]))
    #     w2 = tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2]))
    #     w3 = tf.Variable(tf.random_normal(shape=[n_hidden_2, num_classes]))


        w1 = tf.get_variable(
            'w1', shape=[num_input, n_hidden_1], dtype=tf.float32,
            initializer=tf.random_normal_initializer())
        w2 = tf.get_variable(
            'w2', shape=[n_hidden_1, n_hidden_2], dtype=tf.float32,
            initializer=tf.random_normal_initializer())
        w3 = tf.get_variable(
            'mw3', shape=[n_hidden_2, num_classes], dtype=tf.float32,
            initializer=tf.random_normal_initializer())


        b1 = tf.Variable(tf.zeros([n_hidden_1]))
        b2 = tf.Variable(tf.zeros([n_hidden_2]))
        b3 = tf.Variable(tf.zeros([num_classes]))

        train_size = tf.cast(x_train.shape[0],tf.float32)

        weight_decay = weight_decay_constant/train_size
        # weight_decay = 0.1

        # Create model
        # Hidden fully connected layer with 256 neurons
        z0 = tf.nn.dropout(X, 0.9)
        # z0 = X
        z1 = tf.matmul(z0, w1) + b1
        # z1 = tf.print(z1, z1.get_shape)
        z1_drop = tf.nn.dropout(z1, keep_prob)
        a1 = tf.nn.relu(z1_drop)
        # Hidden fully connected layer with 256 neurons
        z2 = tf.matmul(a1, w2) + b2
        z2_drop = tf.nn.dropout(z2, keep_prob)
        a2 = tf.nn.relu(z2_drop)
        # Output fully connected layer with a neuron for each class
        z3 = tf.matmul(a2, w3) + b3
        logits = z3
        out = tf.nn.softmax(logits)

        # Define loss and optimizer
        losses_op = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y)
        regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
        if is_weight_decay:
            loss_op = tf.reduce_mean(losses_op + weight_decay * regularizer)
        else:
            loss_op = tf.reduce_mean(losses_op)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start training

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            batch_x, batch_y = next_batch(x_train, y_train, batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate})
            # if epoch % display_step == 0 or epoch == 1:
            #     # Calculate batch loss and accuracy for train and val data
            #     loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate})
            #     # print('z1 shape', z1.get_shape)
            #     train_loss.append(loss)
            #     train_acc.append(acc)
            #
            #     batch_val_x, batch_val_y = next_batch(x_val, y_val, batch_size)
            #     loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={X: batch_val_x, Y: batch_val_y, keep_prob:dropout_rate})
            #     val_loss.append(loss_val)
            #     val_acc.append(acc_val)
            #     print('val acc', acc_val)

        save_path = saver.save(sess, "./model_d.ckpt")

        pool_train_loss = train_loss
        pool_val_loss = val_loss
        pool_train_acc = train_acc
        pool_val_acc = val_acc

        # Calculate accuracy for MNIST test images

        if val_mode:
            x_eval = x_val
            y_eval = y_val
        else:
            x_eval = x_test
            y_eval = y_test
        test_predictions_sum = np.zeros(y_eval.shape)
        for t in range(test_iterations):
            test_predictions_t = sess.run(out, feed_dict={X: x_eval,
                                                       Y: y_eval, keep_prob: dropout_rate})
            test_predictions_sum += test_predictions_t
        test_predictions = test_predictions_sum / test_iterations
        test_acc = get_accuracy(test_predictions, y_eval)

        print("Test accuracy Without Acquisition:", test_acc)

        all_accuracy = test_acc


    print('Starting Active Learning in Experiment ', (e+1))

    for i in range(acquisition_iterations):

        # take subset of pool points for Test Time dropout
        # and do acquisition from there
        pool_subset = 10000
        pool_subset_dropout = np.asarray(random.sample(range(0, x_pool.shape[0]), pool_subset))
        x_pool_dropout = x_pool[pool_subset_dropout, :]
        y_pool_dropout = y_pool[pool_subset_dropout]

        # x_pool_dropout = x_pool
        # y_pool_dropout = y_pool

        if random_acquisitions:
            idx = np.arange(0, x_pool_dropout.shape[0])
            x_pool_index = idx[:Queries]
        else:
            # probability score for each class, for each sample in the subset of pool samples for test time dropout
            score_all = np.zeros(shape=(x_pool_dropout.shape[0], num_classes))
            # entropy for each sample in the subset of pool samples for test time dropout
            all_entropy_dropout = np.zeros(shape=x_pool_dropout.shape[0])


            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, "model_d.ckpt")


                for d in range(dropout_iterations):
                    # print('dropout Iteration', d)

                    # batch_x, batch_y = next_batch(x_pool_dropout, y_pool_dropout, batch_size)
                    dropout_score = sess.run(out, feed_dict={X: x_pool_dropout, Y: y_pool_dropout, keep_prob: dropout_rate})
                    score_all = score_all + dropout_score



                    # dropout_score = model.predict_stochastic(x_pool_dropout, batch_size=batch_size, verbose=1)
                    # dropout_score = model.predict_stochastic(x_pool_dropout, batch_size=batch_size, verbose=1)


                    # print('dropout Score', dropout_score)

                    # print(x)

                    score_all = score_all + dropout_score

                    # computing F_x, average uncertainty in the output
                    dropout_score_log = np.log2(dropout_score)
                    entropy_compute = - np.multiply(dropout_score, dropout_score_log)
                    entropy_per_dropout = np.sum(entropy_compute, axis=1)

                    all_entropy_dropout = all_entropy_dropout + entropy_per_dropout

            avg_pi = np.divide(score_all, dropout_iterations)
            log_avg_pi = np.log2(avg_pi)
            entropy_avg_pi = - np.multiply(avg_pi, log_avg_pi)

            # uncertainty of the average output
            entropy_average_pi = np.sum(entropy_avg_pi, axis=1)
            G_x = entropy_average_pi

            # average uncertainty in the output
            average_entropy = np.divide(all_entropy_dropout, dropout_iterations)
            F_x = average_entropy

            # expected information gain
            U_x = G_x - F_x

            # this finds the minimum index
            a_1d = U_x.flatten()
            x_pool_index = a_1d.argsort()[-Queries:]


        # store all the pooled images indexes
        x_pool_all = np.append(x_pool_all, x_pool_index)

        # is_saving pooled images

        # #save only 3 images per iteration
        # for im in range(x_pool_index[0:2].shape[0]):
        # 	Image = x_pool[x_pool_index[im], :, :, :]
        # 	img = Image.reshape((28,28))
        # sp.misc.imsave('/home/ri258/Documents/project/Active-Learning-Deep-Convolutional-Neural-Networks/ConvNets/Cluster_Experiments/dropout_Bald/pooled_Images/' + 'Experiment_' + str(e) + 'pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

        pooled_x = x_pool_dropout[x_pool_index,:]
        pooled_y = y_pool_dropout[x_pool_index]

        # first delete the random subset used for test time dropout from x_pool
        # Delete the pooled point from this pool set (this random subset)
        # then add back the random pool subset with pooled points deleted back to the x_pool set
        delete_pool_x = np.delete(x_pool, (pool_subset_dropout), axis=0)
        delete_pool_y = np.delete(y_pool, (pool_subset_dropout), axis=0)

        delete_pool_x_dropout = np.delete(x_pool_dropout, (x_pool_index), axis=0)
        delete_pool_y_dropout = np.delete(y_pool_dropout, (x_pool_index), axis=0)

        # starts with 40 000, then grows by 2000 each time
        x_pool = np.concatenate((x_pool, x_pool_dropout), axis=0)
        y_pool = np.concatenate((y_pool, y_pool_dropout), axis=0)


        x_train = np.concatenate((x_train, pooled_x), axis=0)
        y_train = np.concatenate((y_train, pooled_y), axis=0)

        # print('x_train', (x_train[0:3,:0:5], x_train.shape))



        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                batch_x, batch_y = next_batch(x_train, y_train, batch_size)
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate})
                if i == acquisition_iterations and (epoch % display_step == 0 or epoch == 1):
                # if (epoch % display_step == 0 or epoch == 1):

                    # Calculate batch loss and accuracy for train and val data
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate})
                    train_loss.append(loss)
                    train_acc.append(acc)

                    batch_val_x, batch_val_y = next_batch(x_val, y_val, batch_size)
                    loss_val, acc_val = sess.run([loss_op, accuracy], feed_dict={X: batch_val_x, Y: batch_val_y, keep_prob: dropout_rate})
                    val_loss.append(loss_val)
                    val_acc.append(acc_val)

                    print('val', acc_val)


            save_path = saver.save(sess, "./model_d.ckpt")

            pool_train_loss = train_loss
            pool_val_loss = val_loss
            pool_train_acc = train_acc
            pool_val_acc = val_acc


            # accumulate the training and valation/test loss after every pooling iteration - for plotting
            pool_val_loss = np.append(pool_val_loss, val_loss)
            pool_train_loss = np.append(pool_train_loss, train_loss)
            pool_val_acc = np.append(pool_val_acc, val_acc)
            pool_train_acc = np.append(pool_train_acc, train_acc)

            if val_mode:
                x_eval = x_val
                y_eval = y_val
            else:
                x_eval = x_test
                y_eval = y_test
            test_predictions_sum = np.zeros(y_eval.shape)
            for t in range(test_iterations):
                test_predictions_t = sess.run(out, feed_dict={X: x_eval,
                                                              Y: y_eval, keep_prob: dropout_rate})
                test_predictions_sum += test_predictions_t
            test_predictions = test_predictions_sum / test_iterations
            test_acc = get_accuracy(test_predictions, y_eval)

            # print('Test accuracy for pooling iteration ', [i+1, acc])
            all_accuracy = np.append(all_accuracy, test_acc)
            print('Test accuracy till now for experiment and pooling iteration', [e+1, i+1, all_accuracy])


    print('Storing accuracy Values for this experiment')
    accuracy_experiments_sum += all_accuracy
    print('Accuracy over all experiments: ', accuracy_experiments_sum/(e+1))

    if is_saving:
        print('Saving Results per Experiment')
        
        if random_acquisitions:
            np.save(
                'results/' + 'dropout_r_train_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_loss)
            np.save(
                'results/' + 'dropout_r_val_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_loss)
            np.save(
                'results/' + 'dropout_r_train_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_acc)
            np.save(
                'results/' + 'dropout_r_val_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_acc)
            np.save(
                'results/' + 'dropout_r_pooled_Image_Index_' + 'Experiment_' + str(
                    e) + '.npy', x_pool_all)
            np.save(
                'results/' + 'dropout_r_accuracy_Results_' + 'Experiment_' + str(
                    e) + '.npy', all_accuracy)
        else:
            np.save(
                'results/' + 'dropout_train_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_loss)
            np.save(
                'results/' + 'dropout_val_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_loss)
            np.save(
                'results/' + 'dropout_train_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_acc)
            np.save(
                'results/' + 'dropout_val_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_acc)
            np.save(
                'results/' + 'dropout_pooled_Image_Index_' + 'Experiment_' + str(
                    e) + '.npy', x_pool_all)
            np.save(
                'results/' + 'dropout_accuracy_Results_' + 'Experiment_' + str(
                    e) + '.npy', all_accuracy)


average_accuracy = np.divide(accuracy_experiments_sum, Experiments)
print('average accuracy Over Experiments:', average_accuracy)

if is_saving:
    print('Saving average accuracy Over Experiments')
    if random_acquisitions:
        np.save(
            'results/' + 'dropout_r_average_accuracy' + '.npy',
            average_accuracy)
    else:
        np.save(
        'results/' + 'dropout_average_accuracy' + '.npy',
        average_accuracy)


time_wholerun = time.time() - time_start

print('total training and testing time:', time_wholerun)
