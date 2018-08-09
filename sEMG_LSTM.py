import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time
from sklearn.model_selection import train_test_split



""" ===================== """
""" ===== FUNCTIONS ===== """
""" ===================== """
def one_hot(labels, n_class=2):
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

    return X_train, X_test



""" ================ """
""" ===== MAIN ===== """
""" ================ """
print('TensorFlow Version\n',tf.VERSION)
print('Python Version\n', sys.version)
print('==>Start the Classification Pipeline using LSTM')
""" ===== Read the matching subjects data from the MATLAB file ===== """
data_mat = sci.loadmat('matching_subjects2.mat')
data = data_mat['DATA']
label = data_mat['LABEL']

num_subject = data.shape[0]
num_channel = data[0, 0].shape[1]
num_rep = np.zeros(num_subject)  # number of repetitions w.r.t
for sub in range(0, num_subject):  # each individual subject
    num_rep[sub] = label[sub, 0].shape[0]

window_size = 8000

print('Total # of Subjects: ', num_subject)
print('Totoal # of Channels: ', num_channel)
print('Sample Size for each Subject\n', num_rep)

""" ===== Leave-One-Out classification experiment ===== """
# 1. Subject selection (TO-DO: Replace with for-loop)
sub_test = 0
print('Testing Subject ', sub_test + 1)

# 2. Generate the testing and training data 
# Assign the testing data array and label array
X_test = np.zeros((int(num_rep[sub_test]), window_size, num_channel))
for ch in range(0, num_channel):
    X_test[:, :, ch] = data[sub_test, 0][0, ch] 

Y_test = label[sub_test, 0] 
Y_test = Y_test.flatten()
Y_test = Y_test.astype(int)
Y_test = one_hot(Y_test)
print('Testing Data Array Size: ', X_test.shape)
print('Label Array Size: ', Y_test.shape)
print('==>Finished obtaining testing data!')

# Assign the training data array and label array
total_rep = num_rep.sum() - num_rep[sub_test]
print('Total # of training repetitions: ', total_rep)

X_train = np.empty((0, window_size, num_channel))
Y_train = np.empty((0, 1))

for sub_train in range(0, num_subject):
    # Exclude the testing subject for the leave-one-out approach
    if sub_train != sub_test:
        print('Loading Training Data From Subject ', sub_train+1)
        x_train = np.zeros((int(num_rep[sub_train]), window_size, num_channel))
        for ch in range(0, num_channel):
            x_train[:, :, ch] = data[sub_train, 0][0, ch]  # store one subject's data

        # Concatenate all subjects signal and labels together
        X_train = np.append(X_train, x_train, axis=0)
        Y_train = np.append(Y_train, label[sub_train, 0], axis=0)

print('Training Data Array Size: ', X_train.shape)
print('Label Array Size: ', Y_train.shape)
print('==>Finished obtaining training data!')
# Convert from the double type of all labels into int type
Y_train = Y_train.flatten()  # flatten the label into a row vector
Y_train = Y_train.astype(int)
Y_train = one_hot(Y_train)

# 3. Split the whole training data into training set and validation set
X_tr, X_vld, Y_tr, Y_vld = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# 4(Optional). Normalize the data
# X_train, X_test = standardize(X_train, X_test)

""" ==================== Deep Learning model ==================== """
print('==>Start building the LSTM Model!')
# Hyperparameters
lstm_size = 12         # 3 times the amount of channels
lstm_layers = 2        # Number of layers
batch_size = 150  # Batch size
seq_len = 8000  # Number of steps
learning_rate = 0.0001
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
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
    # Construct the LSTM inputs and LSTM cells
    lstm_in = tf.transpose(inputs_, [1,0,2]) # reshape into (seq_len, N, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_channels]) # Now (seq_len*N, n_channels)
    
    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None) # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
    
    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)
    
    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state = initial_state)
    
    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')
    
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    #optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping
    
    # Grad clipping
    train_op = tf.train.AdamOptimizer(learning_rate_)

    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

print('==>Finished building the CNN Model!')

""" ==================== Train the LSTM Network =================== """
print('==>Start training the LSTM!')
start = time.time()

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
    
    for e in range(epochs):
        # Initialize 
        state = sess.run(initial_state)
        
        # Loop over batches
        for x,y in get_batches(X_tr, Y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, 
                    initial_state : state, learning_rate_ : learning_rate}
            
            loss, _ , state, acc = sess.run([cost, optimizer, final_state, accuracy], 
                                             feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 25 iterations
            if (iteration%25 == 0):
                
                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                
                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, Y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0, initial_state : val_state}
                    
                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict = feed)
                    
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
    
    saver.save(sess,"checkpoints/sEMG-lstm.ckpt")

end = time.time()

# Plot training and test loss
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 25 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

print('Total Time Spent: ', end-start)
print('==>Finished Training the CNN!')

""" ==================== Test the LSTM Network =================== """
test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    
    for x_t, y_t in get_batches(X_test, Y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1,
                initial_state: test_state}
        
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))