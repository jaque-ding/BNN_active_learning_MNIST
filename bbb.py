from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import range
import numpy as np
import scipy as sp
import random
import math as m  # for pi
import time


random.seed(5820)

gpu = 0
is_batch_test = 0 # needed for gpu

bbb = 1
local_reparametrization = 1

val_mode = 0
random_acquisitions = 0
is_saving = 1

toy_mode = 0

print_first_output = 0
print_train_outputs = 0 # during training
print_probs = 0
# print output during stochastic forward passes for acquisitions
print_output = 0

scale_mixture_prior = 1



Experiments = 3

# Parameters
learning_rate = 0.001
num_epochs = 100  # note: use large number of epochs
batch_size = 120
display_step = 10

# Network Parameters
n_hidden_1 = 400  # 1st layer number of neurons
n_hidden_2 = n_hidden_1 # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

data_size = 55000.0  # divide param loss by this?

sigma_prior = 0.2 #np.exp(-1.0).astype(np.float32)
# for mixture prior
mix_ratio = 0.25
# sigma_prior1 = np.exp(-1.0).astype(np.float32) #0.75
# sigma_prior2 = np.exp(-6.0).astype(np.float32) # 0.2
sigma_prior1 = 0.75
sigma_prior2 = 0.2



# mu_sd = 0.2 #1.0
# rho_init = 0.5
# rho_init_range = 0.2

score = 0
all_accuracy = 0
acquisition_iterations = 98  # 98
test_iterations = 50

# note: use a large number of stochastic iterations
stochastic_iterations = 100  # 100

Queries = 10 # number of samples to add at each acquisition


time_start = time.time()

if toy_mode:

    Experiments = 2
    stochastic_iterations = 2
    test_iterations = 2
    acquisition_iterations = 98
    num_epochs = 2
    is_saving = 0

    n_hidden_1 = 1  # 1st layer number of neurons
    n_hidden_2 = n_hidden_1  # 2nd layer number of neurons

accuracy_experiments_sum = np.zeros(shape=(acquisition_iterations + 1))



for e in range(Experiments):


    print('Experiment Number ', (e + 1 ))


    def one_hot(indices):
        new = np.zeros((indices.size, indices.max() + 1))
        new[np.arange(indices.size), indices] = 1
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

    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    #
    # print('Distribution of training Classes:', np.bincount(y_train))

    # make data into float 32s, divide pixel values by 255, and labels into categorical variables

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_pool = x_pool.astype('float32')
    x_test = x_test.astype('float32')


    train_size = x_train.shape[0]

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

    print('training Model Without Acquisitions in Experiment ', e)


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


        # Store layers weight & bias

        def get_sd(rho):
            return tf.log(1.0 + tf.exp(rho)) # + 1e-5


        # mu_sd = tf.constant(mu_sd)
        # rho_init_range = tf.constant(rho_init_range)

        # same as Recurrent BBB, but causes same activation for each class (0.1) - because sd too small, so all weihgts close to zero?
        w1_mu = tf.get_variable(
            'mu1', shape=[num_input, n_hidden_1], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(num_input)))
        w2_mu = tf.get_variable(
            'mu2', shape=[n_hidden_1, n_hidden_2], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(n_hidden_1)))
        w3_mu = tf.get_variable(
            'mu3', shape=[n_hidden_2, num_classes], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1.0/ np.sqrt(n_hidden_2)))

        w1_rho = tf.get_variable(
            'rho1', shape=[num_input, n_hidden_1], dtype=tf.float32,
            initializer=tf.constant_initializer(np.log(np.exp(1.0 / np.sqrt(num_input)) - 1.0)))
        w2_rho = tf.get_variable(
            'rho2', shape=[n_hidden_1, n_hidden_2], dtype=tf.float32,
            initializer=tf.constant_initializer(np.log(np.exp(1.0 / np.sqrt(n_hidden_1)) - 1.0)))
        w3_rho = tf.get_variable(
            'rho3', shape=[n_hidden_2, num_classes], dtype=tf.float32,
            initializer=tf.constant_initializer(np.log(np.exp(1.0 / np.sqrt(n_hidden_2)) - 1.0)))


        w1_sd = get_sd(w1_rho)
        w2_sd = get_sd(w2_rho)
        w3_sd = get_sd(w3_rho)

        b1 = tf.get_variable(
            'b1', shape=[n_hidden_1,], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        b2 = tf.get_variable(
            'b2', shape=[n_hidden_2,], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        b3 = tf.get_variable(
            'b3', shape=[num_classes,], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        #
        # b1 = tf.Variable(tf.zeros(shape=[n_hidden_1,]))
        # b2 = tf.Variable(tf.zeros(shape=[n_hidden_2,]))
        # b3 = tf.Variable(tf.zeros(shape=[num_classes,]))

        # b1_mu = tf.Variable(tf.random_normal(shape=[n_hidden_1, ], stddev=mu_sd))
        # b2_mu = tf.Variable(tf.random_normal(shape=[n_hidden_2, ], stddev=mu_sd))
        # b3_mu = tf.Variable(tf.random_normal(shape=[num_classes, ], stddev=mu_sd))
        # b1_rho = tf.Variable(
        #     tf.random_uniform(shape=[n_hidden_1, ], minval=-rho_init_range / 2.0, maxval=rho_init_range))
        # b2_rho = tf.Variable(
        #     tf.random_uniform(shape=[n_hidden_2, ], minval=-rho_init_range / 2.0, maxval=rho_init_range))
        # b3_rho = tf.Variable(
        #     tf.random_uniform(shape=[num_classes, ], minval=-rho_init_range / 2.0, maxval=rho_init_range))

        eps_w1 = (tf.random_normal(shape=[num_input, n_hidden_1], stddev=1.0))
        eps_w2 = (tf.random_normal(shape=[n_hidden_1, n_hidden_2], stddev=1.0))
        eps_w3 = (tf.random_normal(shape=[n_hidden_2, num_classes], stddev=1.0))
        # eps_b1 = (tf.random_normal(shape=[n_hidden_1, ], stddev=1.0))
        # eps_b2 = (tf.random_normal(shape=[n_hidden_2, ], stddev=1.0))
        # eps_b3 = (tf.random_normal(shape=[num_classes, ], stddev=1.0))

        w1 = w1_mu + w1_sd * eps_w1
        w2 = w2_mu + w2_sd * eps_w2
        w3 = w3_mu + w3_sd * eps_w3

        # b1 = tf.add(b1_mu, tf.multiply(tf.log(tf.add(tf.ones_like(b1_rho, dtype=tf.float32), tf.exp(b1_rho))), eps_b1))
        # b2 = tf.add(b2_mu, tf.multiply(tf.log(tf.add(tf.ones_like(b2_rho, dtype=tf.float32), tf.exp(b2_rho))), eps_b2))
        # b3 = tf.add(b3_mu, tf.multiply(tf.log(tf.add(tf.ones_like(b3_rho, dtype=tf.float32), tf.exp(b3_rho))), eps_b3))

        # local reparametrization
        def get_z(inp, w_mu, w_sd, eps, eps_length):
            # rand_n_like?
            eps = tf.random_normal([tf.shape(inp)[0], eps_length], stddev=1.0)
            z = tf.matmul(inp, w_mu) + tf.sqrt(tf.matmul(tf.square(inp), tf.square(w_sd))) * eps
            return z

        # Create model

        if local_reparametrization:
            z1 = get_z(X, w1_mu, w1_sd, eps_w1, n_hidden_1) + b1
            a1 = tf.nn.relu(z1)
            z2 = get_z(a1, w2_mu, w2_sd, eps_w2, n_hidden_2) + b2
            a2 = tf.nn.relu(z2)
            z3 = get_z(a2, w3_mu, w3_sd, eps_w3, num_classes) + b3
            logits = z3
        else:
            z1 = tf.matmul(X, w1) + b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1, w2) + b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2, w3) + b3
            logits = z3


        # Construct model
        out = tf.nn.softmax(logits)

        # Define loss and optimizer

        def gaussian(x, mu, sigma):
            # returns density for x, given mu and sigma.
            a = tf.reduce_max(x,axis=0) # for log sum ex
            return tf.exp(- tf.square(x - mu) / (2.0 * tf.square(sigma)) - a) / (
                    tf.sqrt(2.0 * tf.constant(m.pi)) * tf.square(sigma))

        def log_gaussian(x, mu, sigma):
            # returns log gaussian density for x, given mu and sigma.
            return - tf.square(x - mu) / (2.0 * tf.square(sigma)) - tf.log( tf.sqrt(2.0 * tf.constant(m.pi))*sigma)


        def mix_gaussian(x, mu, sigma_prior1, sigma_prior2, mix_ratio):
            gaussian1 = mix_ratio * gaussian(x, mu, sigma_prior1)
            gaussian2 = (1.0 - mix_ratio) * gaussian(x, mu, sigma_prior2)
            return gaussian1 + gaussian2


        def log_q(x, mu, sigma):
            # q = tf.clip_by_value(gaussian(x, mu, sigma), 1e-10, 1.0)
            # q = gaussian(x, mu, sigma)
            logq = log_gaussian(x, mu, sigma)
            return tf.reduce_sum(logq)

        def log_p_sm(x, mu, sigma_prior1, sigma_prior2, mix_ratio):
            p = tf.clip_by_value(mix_gaussian(x, mu, sigma_prior1, sigma_prior2, mix_ratio), 1e-10, 1.0)
            # p = mix_gaussian(x, mu, sigma_prior1, sigma_prior2, mix_ratio)
            a = tf.reduce_max(x,axis=0)
            logp = tf.log(p) + a
            return tf.reduce_sum(logp)

        def log_p(x, mu, sigma_prior1, sigma_prior2, mix_ratio):
            logp = log_gaussian(x, mu, sigma)
            return tf.reduce_sum(logp)



        log_likelihood_sum = - tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        # log_likelihood_mean = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=logits, labels=Y))
        q1 = log_q(w1, w1_mu, get_sd(w1_rho))
        q2 = log_q(w2, w2_mu, get_sd(w2_rho))
        q3 = log_q(w3, w3_mu, get_sd(w3_rho))
        q = q1 + q2 + q3

        if scale_mixture_prior:
            p1 = log_p_sm(w1, 0.0, sigma_prior1, sigma_prior2, mix_ratio)
            p2 = log_p_sm(w2, 0.0, sigma_prior1, sigma_prior2, mix_ratio)
            p3 = log_p_sm(w3, 0.0, sigma_prior1, sigma_prior2, mix_ratio)
        else:
            p1 = log_q(w1, 0.0, sigma_prior)
            p2 = log_q(w2, 0.0, sigma_prior)
            p3 = log_q(w3, 0.0, sigma_prior)

        p = p1 + p2 + p3

        real_batch_size = tf.cast(tf.shape(X)[0], tf.float32)

        # summed
        loss_op = (q - p - log_likelihood_sum) / real_batch_size * train_size


        # # mean
        # loss_op = (q - p) / train_size - log_likelihood_mean

        # wrong
        # loss_op = (q - p - log_likelihood_sum) / data_size * train_size



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
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            # if print_first_output:
            #     if epoch == 0:
            #         a1 = sess.run([out], feed_dict={X: batch_x, Y: batch_y})
            #         print('activity', a1)
            # if epoch % display_step == 0 or epoch == 1:
            #     # Calculate batch loss and accuracy for train and val data
            #     loss, acc, post, prior, log_lik= sess.run([loss_op, accuracy, q, p, log_likelihood_sum],
            #                                                feed_dict={X: batch_x, Y: batch_y})
            #     if print_train_outputs:
            #         a1 = sess.run([out], feed_dict={X: batch_x, Y: batch_y})
            #         print('activity', a1)
            #
            #     # print('z1', z1)
            #     if print_probs:
            #         print('post, prior, lok lik', (post/data_size, prior/data_size, log_lik))
            #     train_loss.append(loss)
            #     train_acc.append(acc)
            #
            #     batch_val_x, batch_val_y = next_batch(x_val, y_val, batch_size)
            #     loss_val, acc_val = sess.run([loss_op, accuracy],
            #                                  feed_dict={X: batch_val_x, Y: batch_val_y})
            #     val_loss.append(loss_val)
            #     val_acc.append(acc_val)

        save_path = saver.save(sess, "./model_b.ckpt")

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
            if is_batch_test:
                test_predictions_t = np.zeros((1,10))
                for i in range(50):
                    test_x, test_y = mnist.test.next_batch(200)
                    test_predictions_batch = sess.run(out, feed_dict={X: test_x, Y: test_y})
                    # print('shape should be 200x10', test_predictions_batch.shape)
                    test_predictions_t = np.concatenate((test_predictions_t, test_predictions_batch), axis = 0)
                    # print('shape should be 201 401 etc x10', test_predictions_t.shape)

                # print('shape should be 10001', test_predictions_t.shape)

                test_predictions_t = np.delete(test_predictions_t, (0), axis=0) # delete first row of zeros

            else:
                test_predictions_t = sess.run(out, feed_dict={X: x_eval,
                                                       Y: y_eval})
            # print('shape should be 10000x10', test_predictions_t.shape)
            # print('first row', test_predictions_t[0,:])
            # print('last row', test_predictions_t[-1,:])


            test_predictions_sum += test_predictions_t
        test_predictions = test_predictions_sum / test_iterations
        test_acc = get_accuracy(test_predictions, y_eval)
        print("Test accuracy Without Acquisition:", test_acc)

        all_accuracy = test_acc

    print('Starting Active Learning in Experiment ', e)

    for i in range(acquisition_iterations):

        # take subset of pool points for Test Time stochastic
        # and do acquisition from there
        pool_subset = 10000
        pool_subset_stochastic = np.asarray(random.sample(range(0, x_pool.shape[0]), pool_subset))
        x_pool_stochastic = x_pool[pool_subset_stochastic, :]
        y_pool_stochastic = y_pool[pool_subset_stochastic]

        # x_pool_stochastic = x_pool
        # y_pool_stochastic = y_pool

        if random_acquisitions:
            idx = np.arange(0,x_pool_stochastic.shape[0])
            x_pool_index = idx[:Queries]
        else:
            # probability score for each class, for each sample in the subset of pool samples for test time stochastic
            score_all = np.zeros(shape=(x_pool_stochastic.shape[0], num_classes))
            # entropy for each sample in the subset of pool samples for test time stochastic
            all_entropy_stochastic = np.zeros(shape=x_pool_stochastic.shape[0])

            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, "model_b.ckpt")

                for d in range(stochastic_iterations):
                    # print('stochastic Iteration', d)

                    # batch_x, batch_y = next_batch(x_pool_stochastic, y_pool_stochastic, batch_size)
                    stochastic_score = sess.run(out, feed_dict={X: x_pool_stochastic, Y: y_pool_stochastic})

                    # if print_output:
                    #     print('stochastic Score', np.round(stochastic_score, decimals=1))

                    # print(x)

                    score_all = score_all + stochastic_score

                    # computing F_x, average uncertainty in the output
                    stochastic_score_log = np.log2(stochastic_score)
                    entropy_compute = - np.multiply(stochastic_score, stochastic_score_log)
                    entropy_per_stochastic = np.sum(entropy_compute, axis=1)

                    all_entropy_stochastic = all_entropy_stochastic + entropy_per_stochastic

            avg_pi = np.divide(score_all, stochastic_iterations)
            # if print_output:
            #     print('average Score', (np.round(avg_pi, decimals=1), np.sum(avg_pi, axis=1)))

            # print('average score and sum', (avg_pi[4,:], np.sum(avg_pi, axis=1)))
            log_avg_pi = np.log2(avg_pi)
            entropy_avg_pi = - np.multiply(avg_pi, log_avg_pi)

            # uncertainty of the average output
            entropy_average_pi = np.sum(entropy_avg_pi, axis=1)
            G_x = entropy_average_pi

            # average uncertainty in the output
            average_entropy = np.divide(all_entropy_stochastic, stochastic_iterations)
            F_x = average_entropy

            # expected information gain
            U_x = G_x - F_x
            # if print_output:
            #     print('U, G, F', (U_x, F_x, G_x))
            #     print('shape', U_x.shape)

            # this finds the indices of the largest
            a_1d = U_x.flatten()
            # x_pool_index = a_1d.argsort()[-Queries:]
            x_pool_index = a_1d.argsort()[0:Queries]




        # store all the pooled images indexes
        x_pool_all = np.append(x_pool_all, x_pool_index)

        pooled_x = x_pool_stochastic[x_pool_index, :]
        pooled_y = y_pool_stochastic[x_pool_index]

        # first delete the random subset used for test time stochastic from x_pool
        # Delete the pooled point from this pool set (this random subset)
        # then add back the random pool subset with pooled points deleted back to the x_pool set
        delete_pool_x = np.delete(x_pool, (pool_subset_stochastic), axis=0)
        delete_pool_y = np.delete(y_pool, (pool_subset_stochastic), axis=0)

        delete_pool_x_stochastic = np.delete(x_pool_stochastic, (x_pool_index), axis=0)
        delete_pool_y_stochastic = np.delete(y_pool_stochastic, (x_pool_index), axis=0)

        # starts with 40 000, then grows by 2000 each time
        x_pool = np.concatenate((x_pool, x_pool_stochastic), axis=0)
        y_pool = np.concatenate((y_pool, y_pool_stochastic), axis=0)

        x_train = np.concatenate((x_train, pooled_x), axis=0)
        y_train = np.concatenate((y_train, pooled_y), axis=0)

        train_size = x_train.shape[0]



        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                batch_x, batch_y = next_batch(x_train, y_train, batch_size)
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if i == acquisition_iterations and (epoch % display_step == 0 or epoch == 1):
                    # Calculate batch loss and accuracy for train and val data
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    train_loss.append(loss)
                    train_acc.append(acc)

                    batch_val_x, batch_val_y = next_batch(x_val, y_val, batch_size)
                    loss_val, acc_val = sess.run([loss_op, accuracy],
                                                 feed_dict={X: batch_val_x, Y: batch_val_y})
                    val_loss.append(loss_val)
                    val_acc.append(acc_val)

            save_path = saver.save(sess, "./model_b.ckpt")

            pool_train_loss = train_loss
            pool_val_loss = val_loss
            pool_train_acc = train_acc
            pool_val_acc = val_acc

            # accumulate the training and validation/test loss after every pooling iteration - for plotting
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
                                                              Y: y_eval})
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
                'results/' + 'bbb_r_train_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_loss)
            np.save(
                'results/' + 'bbb_r_val_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_loss)
            np.save(
                'results/' + 'bbb_r_train_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_acc)
            np.save(
                'results/' + 'bbb_r_val_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_acc)
            np.save(
                'results/' + 'bbb_r_pooled_Image_Index_' + 'Experiment_' + str(
                    e) + '.npy', x_pool_all)
            np.save(
                'results/' + 'bbb_r_accuracy_Results_' + 'Experiment_' + str(
                    e) + '.npy', all_accuracy)
        else:
            np.save(
                'results/' + 'bbb_train_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_loss)
            np.save(
                'results/' + 'bbb_val_loss_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_loss)
            np.save(
                'results/' + 'bbb_train_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_train_acc)
            np.save(
                'results/' + 'bbb_val_acc_' + 'Experiment_' + str(
                    e) + '.npy', pool_val_acc)
            np.save(
                'results/' + 'bbb_pooled_Image_Index_' + 'Experiment_' + str(
                    e) + '.npy', x_pool_all)
            np.save(
                'results/' + 'bbb_accuracy_Results_' + 'Experiment_' + str(
                    e) + '.npy', all_accuracy)


average_accuracy = np.divide(accuracy_experiments_sum, Experiments)
print('Average accuracy Over Experiments:', average_accuracy)
if is_saving:
    print('Saving average accuracy Over Experiments')
    if random_acquisitions:
        np.save(
            'results/' + 'bbb_r_average_accuracy' + '.npy',
            average_accuracy)
    else:
        np.save(
        'results/' + 'bbb_average_accuracy' + '.npy',
        average_accuracy)

time_wholerun = time.time() - time_start

print('total training and testing time:', time_wholerun)

