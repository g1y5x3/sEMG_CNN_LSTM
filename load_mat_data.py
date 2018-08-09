import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

# Read the matching subjects data from the MATLAB file
data_mat = sci.loadmat('matching_subjects.mat')
data_healthy = data_mat['DATA_HEALTHY'] 
data_fatigue = data_mat['DATA_DYSFUNC']

num_healthy = data_healthy.shape[0];
num_fatigue = data_fatigue.shape[0];

# Create vector that stores the # of repetitons for each subject
size_window = 8000 

size_healthy = np.zeros((num_healthy,1))
for sub in range(0,num_healthy):
    size_tmp = data_healthy[sub,0][0,0].shape
    size_healthy[sub,0] = size_tmp[0]
print('Healthy Subjects Data Size:\n', size_healthy)

size_fatigue = np.zeros((num_fatigue,1))
for sub in range(0,num_fatigue):
    size_tmp = data_fatigue[sub,0][0,0].shape
    size_fatigue[sub,0] = size_tmp[0]
print('Fatigue Subjects Data Size:\n', size_fatigue)

# Generate the training and testing data based on the 
# Leave-One-Out approach

# print(data_healthy.shape[0])
# # Create the data array for the sEMG signals
# print('Subject 1 Channel1 Size: ', data_healthy[0,0][0,0].shape)
# 
# sample_signal = data_healthy[0,0][0,0][0,:]
# plt.plot(sample_signal)
# plt.show()
