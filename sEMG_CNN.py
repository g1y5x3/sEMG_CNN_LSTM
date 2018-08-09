import numpy as np
import scipy.io as sci
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time
from sklearn.model_selection import train_test_split



""" =============================================================== """
""" ========================== FUNCTIONS ========================== """
""" =============================================================== """
def one_hot(labels, n_class=2):
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size=50):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

# def standardize(train, test):
#     """ Standardize data """
# 
#     # Standardize train and test
#     X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
#     X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]
# 
#     return X_train, X_test

def standardize(X):
    """ Standardize data """

    # Standardize train and test
    Y = (X- np.mean(X, axis=0)[None, :, :]) / np.std(X, axis=0)[None, :, :]

    return Y




""" ========================================================== """
""" ========================== MAIN ========================== """
""" ========================================================== """
print('TensorFlow Version\n',tf.VERSION)
print('Python Version\n', sys.version)
print('==>Start the Classification Pipeline using CNN')

""" ===== Read the matching subjects data from the MATLAB file ===== """
data_mat = sci.loadmat('matching_subjects1.mat')
data = data_mat['DATA']
label = data_mat['LABEL']

num_subject = data.shape[0]
num_channel = data[0, 0].shape[1]
num_rep = np.zeros(num_subject)  # number of repetitions w.r.t
for sub in range(0, num_subject):  # each individual subject
    num_rep[sub] = label[sub, 0].shape[0]

window_size = 8000

# Save all the log infos into a text file
f = open('sEMG_CNN_rawdata_6conv_standardize_relu.txt', 'w')

print('Total # of Subjects: {}\n'.format(num_subject))
print('Totoal # of Channels: {}\n'.format(num_channel))
print('# of Data Samples for each Subject {}\n'.format(num_rep))

f.write('Total # of Subjects: {}\n'.format(num_subject))
f.write('Totoal # of Channels: {}\n'.format(num_channel))
f.write('# of Data Samples for each Subject {}\n'.format(num_rep))

""" ===== Leave-One-Out classification experiment ===== """
accuracy_train = np.zeros(num_subject)
accuracy_validation = np.zeros(num_subject)
accuracy_test = np.zeros(num_subject)

# 1. Subject selection
start_all = time.time()     # measure the total time of the whole experiment
for sub_test in range(0,num_subject):
# for sub_test in range(0,2):
    print('==============================\n')
    f.write('==============================\n')

    print('==>Loading testing data......')
    f.write('==>Loading testing data......\n')

    print('Testing Subject {}'.format(sub_test + 1))
    f.write('Testing Subject {}\n'.format(sub_test + 1))

    save_title = "sEMG_%d.ckpt" % (sub_test+1)
    save_path = "checkpoints-cnn/" + save_title
    
    # 2. Generate the testing and training data 
    # Assign the testing data array and label array
    X_test = np.zeros((int(num_rep[sub_test]), window_size, num_channel))
    for ch in range(0, num_channel):
        X_test[:, :, ch] = data[sub_test, 0][0, ch]
    
    Y_test = label[sub_test, 0] 

    # (Optional) Standardize the data
    X_test = standardize(X_test)

    # Convert the label as the type that can be accepted by CNN
    Y_test = Y_test.flatten()
    Y_test = Y_test.astype(int)
    Y_test = one_hot(Y_test)

    # print the information about each testing subjects
    print('Testing Data Array Size: ', X_test.shape)
    print('Label Array Size: ', Y_test.shape)
    print('==>Finished obtaining testing data!')
    f.write('Testing Data Array Size: {}\n'.format(X_test.shape))
    f.write('Label Array Size: {}\n'.format(Y_test.shape))
    f.write('==>Finished obtaining testing data!\n')

    # Assign the training data array and label array
    print('==>Loading training data......')
    f.write('==>Loading training data......\n')

    total_rep = num_rep.sum() - num_rep[sub_test]
    print('Total # of training repetitions: ', total_rep)
    f.write('Total # of training repetitions: {}\n'.format(total_rep))

    X_train = np.empty((0, window_size, num_channel))
    Y_train = np.empty((0, 2))

    for sub_train in range(0, num_subject):
        # Exclude the testing subject for the leave-one-out approach
        if sub_train != sub_test:
            print('Loading Training Data From Subject ', sub_train+1)

            x_train = np.zeros((int(num_rep[sub_train]), window_size, num_channel))
            for ch in range(0, num_channel):
                x_train[:, :, ch] = data[sub_train, 0][0, ch]  # store one subject's data

            # Convert from the double type of all labels into int type
            y_train = label[sub_train, 0]
            y_train = y_train.flatten()
            y_train = y_train.astype(int)
            y_train = one_hot(y_train)

            # (Optional) Standardize the data
            x_train = standardize(x_train)

            # Concatenate all subjects signal and labels together
            X_train = np.append(X_train, x_train, axis=0)
            Y_train = np.append(Y_train, y_train, axis=0)

    print('Training Data Array Size: ', X_train.shape)
    print('Label Array Size: ', Y_train.shape)
    print('==>Finished obtaining training data!')

    f.write('Training Data Array Size: {}\n'.format(X_train.shape))
    f.write('Label Array Size: {}\n'.format(Y_train.shape))
    f.write('==>Finished obtaining training data!\n')

    # 3. Split the whole training data into training set and validation set
    X_tr, X_vld, Y_tr, Y_vld = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    # 4(Optional). Normalize the data
    # X_train, X_test = standardize(X_train, X_test)

    """ ==================== Deep Learning model ==================== """
    print('==>Start building the CNN Model!')
    f.write('==>Start building the CNN Model!\n')

    # Hyperparameters
    batch_size = 25  # Batch size
    seq_len = 8000  # Number of steps
    learning_rate = 0.00005
    epochs = 500

    print('Batch Size: ', batch_size)
    print('Learning Rate: ', learning_rate)
    print('Total # of Epochs: ', epochs)

    n_classes = 2
    n_channels = 4

    # Construct the graph
    # placeholders
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    # Build Convolutional Layers
    with graph.as_default():
        # (batch, 8000, 4) --> (batch, 4000, 8)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=8, kernel_size=2, strides=1,
                               padding='same', activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

       # (batch, 4000, 8) --> (batch, 2000, 16)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=16, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
 
        # (batch, 2000, 16) --> (batch, 1000, 32)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=32, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        # (batch, 1000, 32) --> (batch, 500, 64)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=64, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        # (batch, 500, 64) --> (batch, 250, 128)
        conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=128, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

        # (batch, 250, 128) --> (batch, 125, 256)
        conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=256, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
        max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same')


    # Flatten and pass to the classifier
    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_6, (-1, 32000))
        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

        # Predictions
        logits = tf.layers.dense(flat, n_classes)

        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    print('==>Finished building the CNN Model!')
    f.write('==>Finished building the CNN Model!\n')

    """ ==================== Train the CNN Network =================== """
    print('==>Start training the CNN!')
    f.write('==>Start training the CNN!\n')

    start_ind = time.time()

    if not os.path.exists('checkpoints-cnn'):
        os.mkdir('checkpoints-cnn')

    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1

        # Loop over epochs
        for e in range(epochs):

            # Loop over batches
            for x, y in get_batches(X_tr, Y_tr, batch_size):

                # Feed dictionary
                feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: learning_rate}

                # Loss
                loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 500 iters
                if (iteration % 500 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Train loss: {:6f}".format(loss),
                        "Train acc: {:.6f}".format(acc))

                # Compute validation loss at every 100 iterations
                if (iteration % 1000 == 0):
                    val_acc_ = []
                    val_loss_ = []

                    for x_v, y_v in get_batches(X_vld, Y_vld, batch_size):
                        # Feed
                        feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0}

                        # Loss
                        loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)

                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Validation loss: {:6f}".format(np.mean(val_loss_)),
                        "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1
        
        end_ind = time.time()

        accuracy_train[sub_test] = acc
        accuracy_validation[sub_test] = acc_v

        save_title = "sEMG_%d.ckpt" % (sub_test+1)
        save_path = "checkpoints-cnn/" + save_title
        saver.save(sess, save_path)

    # # Plot training and test loss
    # t = np.arange(iteration-1)
    # 
    # plt.figure(figsize = (6,6))
    # plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Loss")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    # 
    # # Plot Accuracies
    # plt.figure(figsize = (6,6))
    # 
    # plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Accuray")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    print('Total Time Spent: ', end_ind-start_ind)
    print('==>Finished Training the CNN!')
    f.write('Total Time Spent: {}\n'.format(end_ind-start_ind))
    f.write('==>Finished Training the CNN!\n')

    """ ==================== Test the CNN Network =================== """
    test_acc = []

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
        
        for x_t, y_t in get_batches(X_test, Y_test, batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}
            
            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)

        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
        f.write("Test accuracy: {:.6f}\n".format(np.mean(test_acc)))

        accuracy_test[sub_test] = np.mean(test_acc)
    
end_all = time.time()

""" ==================== Output the final results ==================== """
print('\n')
print('Experiments Results:')
print('Testing Accuracy: {}'.format(accuracy_test))
print("Average Training Accuracy: {:.6f}".format(np.mean(accuracy_train)))
print("Average Validation Accuracy: {:.6f}".format(np.mean(accuracy_validation)))
print("Average Testing Accuracy: {:.6f}".format(np.mean(accuracy_test)))
print('Sensitivity: {}'.format(np.mean(accuracy_test[0:8])))
print('Specificity: {}'.format(np.mean(accuracy_test[10:18])))
print("Total Time Spend: ", end_all-start_all)

f.write('\n')
f.write('Experiments Results:\n')
f.write('Testing Accuracy: {}\n'.format(accuracy_test))
f.write("Average Training Accuracy: {:.6f}\n".format(np.mean(accuracy_train)))
f.write("Average Validation Accuracy: {:.6f}\n".format(np.mean(accuracy_validation)))
f.write("Average Testing Accuracy: {:.6f}\n".format(np.mean(accuracy_test)))

f.write("Total Time Spend: {}\n".format(end_all-start_all))
