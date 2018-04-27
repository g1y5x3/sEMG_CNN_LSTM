import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

# Read the matching subjects data from the MATLAB file
data_mat = sci.loadmat('matching_subjects2.mat')
data  = data_mat['DATA'] 
label = data_mat['LABEL']

num_subject = data.shape[0]
num_channel = data[0,0].shape[1]
num_rep = np.zeros(num_subject)         # number of repetitions w.r.t
for sub in range(0,num_subject):            # each individual subject
    num_rep[sub] = label[sub,0].shape[0]

# print(data.shape)
# print(label.shape)
# print(data[0,0].shape)
# print(label[0,0].shape)
# print(label[0,0].ndim)

print('Total # of Subjects: ', num_subject)
print('Totoal # of Channels: ', num_channel)
print('Sample Size:\n', num_rep)

# Leave-One-Out classification experiment
sub_test = 0
print('Testing Subject ', sub_test+1)

# Generate the training data 
# Calculate the total repetitons of training samples
window_size = 8000
total_rep = num_rep.sum() - num_rep[sub_test]

# print(num_rep.sum())
# print(num_rep[sub_test])
print('Total # of training repetitions: ', total_rep)

# Assign the training data vector and its labels
X_tr = np.empty((0, window_size, num_channel))
Y_tr = np.empty((0, 1))
# print(X_tr.shape)
# print(X_tr.ndim)
# print(Y_tr.shape)
# print(Y_tr.ndim)
# X_tr = []

for sub_train in range(0,num_subject):
    # Exclude the testing subject for the leave-one-out approach
    if sub_train != sub_test:
        x_tr = np.zeros((int(num_rep[sub_train]), window_size, num_channel))
        for ch in range(0,num_channel):
            x_tr[:,:,ch] = data[sub_train,0][0,ch]  # store one subject's data
        
        # Concatenate all subjects signal and labels together
        X_tr = np.append(X_tr, x_tr, axis=0)
        Y_tr = np.append(Y_tr, label[sub_train,0], axis=0)

print(X_tr.shape)
print(Y_tr.shape)

sample_signal = X_tr[1,:,1] 
plt.plot(sample_signal)
plt.show()

print('Obtained the training vector!\n')

