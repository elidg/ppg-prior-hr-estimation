# Script to test BeliefPPG on VT dataset

import os
import numpy as np
from scipy.io import savemat
from beliefppg import infer_hr_uncertainty
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Analyzed subject settings
folderData = './BeliefPPG_Data/'
folderDS = 'MAX-HIE/' # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
subject_id = 'sub7' # for dataset MAX-HIE use 'subX'; for dataset SUBMAX-HIE use 'AXXX' or 'BXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"
session_id = 'visit1' # for dataset MAX-HIE use 'visit1'; for dataset SUBMAX-HIE use 'MTXXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"

# Signal parameters
if 'SUBMAX-HIE' in folderDS:
	 # SUBMAX-HIE
	ppg_sampling_rate = 176  # Hz (sampling rate of ppg sensor) #250 for VT dataset; #176 for newly collected data
	acc_sampling_rate = 416 # Hz (sampling rate of accelerometer) #250 for VT dataset; #416 for newly collected data
else:
	# MAX-HIE
	ppg_sampling_rate = 250  # Hz (sampling rate of ppg sensor) #250 for VT dataset; #176 for newly collected data
	acc_sampling_rate = 250 # Hz (sampling rate of accelerometer) #250 for VT dataset; #416 for newly collected data
	subjects_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

folderResults = './Results/' + folderDS + subject_id + '_' + session_id # change to './Results/' + folderDS if you want the data in one folder and add subject_id + '_' + session_id to the file names to save (.mat and .png) 
os.makedirs(folderResults, exist_ok=True)

# for subject_n in subjects_all: # uncomment this if you want to run the full MAX-HIE

# 	subject = 'sub' + str(subject_n) # uncomment this if you want to run the full MAX-HIE
    
print('Analyzing Subject ' + subject_id + ' Session ' + session_id)

# Load data item containing the PPG, IMU signals (ACC in this case), and ground-truth HR (from ECG in this case)
data = np.load(folderData + folderDS + subject_id + '_' + session_id + '.npy', allow_pickle=True).item()

t = data['t']
ppg = data['PPG head'].reshape((-1,1)) # reshape ppg to (n_samples, n_channels)
IMU_X = data['IMU X head']
IMU_Y = data['IMU Y head']
IMU_Z = data['IMU Z head']
acc = np.stack([IMU_X,IMU_X, IMU_Z], axis=-1)

GT_hr = data['ground truth HR'] # derived from peak-to-peak RR intervals for MAX-HIE and subject B023 of SUBMAX-HIE, and Hexoskin HR every 1s for other subjects of SUBMAX-HIE
t_GT_hr = data['t ground truth HR']

# BeliefPPG HR estimation using PPG+ACC
hr, uncertainty, time_intervals = infer_hr_uncertainty(ppg=ppg, ppg_freq=ppg_sampling_rate, acc=acc, acc_freq=acc_sampling_rate)
midpoint_idxs = (np.mean(time_intervals, axis=-1)*ppg_sampling_rate).astype(int)
t_hr = midpoint_idxs/ppg_sampling_rate

# Compute mean ground truth HR every 2s as BeliefPPG
mean_GT_hr = []
for start, end in time_intervals:
	# Find GT_hr values within this time interval
	mask = (t_GT_hr >= start) & (t_GT_hr < end)
	interval_hr = GT_hr[mask]

	if len(interval_hr) > 0:
		mean_GT_hr.append(np.mean(interval_hr))
	else:
		mean_GT_hr.append(0)  # or 0 if you prefer

mae_hr = mean_absolute_error(mean_GT_hr,hr)
print("HR-PPG with ACC vs Ground-truth HR")
print(f"Mean Absolute Error (MAE): {mae_hr}")

# BeliefPPG HR estimation using only PPG
hr_wo_acc, uncertainty_wo_acc, time_intervals_wo_acc = infer_hr_uncertainty(ppg=ppg, ppg_freq=ppg_sampling_rate, acc=None, acc_freq=None) # infer HR without using the accelerometer
midpoint_idxs_wo_acc = (np.mean(time_intervals_wo_acc, axis=-1)*ppg_sampling_rate).astype(int)
t_hr_wo_acc = midpoint_idxs_wo_acc/ppg_sampling_rate

mae_hr_wo_acc = mean_absolute_error(mean_GT_hr,hr_wo_acc)
print("HR-PPG without ACC vs Ground-truth HR")
print(f"Mean Absolute Error (MAE): {mae_hr_wo_acc}")

plt.figure(figsize=(16,9))
plt.plot(t_hr, hr, c='C0', label='Estimated HR')
plt.plot(t_GT_hr, GT_hr, c='C2', label='Ground Truth HR')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate [bpm]')
plt.legend()
plt.xlim((0, np.max(t_hr)))
plt.title('BeliefPPG - Estimated HR with Accelerometer')
if not os.path.isfile(folderResults + "/beliefppg_estimated_hr.png"): # add subject_id + '_' + session_id if you want to save the data in the same folder
	plt.savefig(folderResults + "/beliefppg_estimated_hr.png",dpi=180) # add subject_id + '_' + session_id if you want to save the data in the same folder
	plt.close()

plt.figure(figsize=(16,9))
plt.plot(t_hr_wo_acc, hr_wo_acc, c='C0', label='Estimated HR wo. Accelerometer')
plt.plot(t_GT_hr, GT_hr, c='C2', label='Ground Truth HR')   
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate [bpm]')
plt.legend()
plt.xlim((0, np.max(t_hr)))
plt.title('BeliefPPG - Estimated HR without Accelerometer')
if not os.path.isfile(folderResults + "/beliefppg_estimated_hr_wo_acc.png"): # add subject_id + '_' + session_id if you want to save the data in the same folder
	plt.savefig(folderResults + "/beliefppg_estimated_hr_wo_acc.png",dpi=180) # add subject_id + '_' + session_id if you want to save the data in the same folder
	plt.close()

# Save .mat file with estimated HR results
data_to_save = {
	't_hr': t_hr,                   
	'hr_est': hr,  
}

filename_mat = folderResults + "/beliefppg_estimated_hr.mat" # add subject_id + '_' + session_id if you want to save the data in the same folder
savemat(filename_mat, data_to_save)

# Save .mat file with estimated HR results
data_to_save = {
	't_hr_wo_acc': t_hr_wo_acc,                   
	'hr_est_wo_acc': hr_wo_acc,  
}

filename_mat = folderResults + "/beliefppg_estimated_hr_wo_acc.mat" # add subject_id + '_' + session_id if you want to save the data in the same folder
savemat(filename_mat, data_to_save)
	