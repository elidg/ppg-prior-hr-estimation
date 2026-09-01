# Building prior distribution of validated HR series from a population performing a maximal test for PPG-exclusive HR estimation with Dynamic time warping (DTW)
import os
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.barycenters import softdtw_barycenter
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# Folder for prior dataset
folderDSprior = "./HR_Validation/MAX-HIE/" # available dataset to compute prior: 'MAX-HIE' and 'ACTES'. Use these strings to name the folders. For 'ACTES', download it from physionet.org/content/actes-cycloergometer-exercise/1.0.0/ and save it in 'HR_Validation/ACTES/'

# Analyzed subject settings
folderHRval = "./HR_Validation/MAX-HIE/" # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
folderDS = 'MAX-HIE/' # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
subject_analyzed = 'sub7' # for dataset MAX-HIE use 'subX'; for dataset SUBMAX-HIE use 'AXXX' or 'BXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"
session_id = 'visit1' # for dataset MAX-HIE use 'visit1'; for dataset SUBMAX-HIE use 'MTXXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"

# HR prior parameters
freqs = [40.28,41.20,42.11,43.03,43.95,44.86,45.78,46.69,47.61,48.52,49.44,50.35,51.27,52.19,53.10,54.02,54.93,55.85,56.76,57.68,58.59,59.51,60.42,61.34,62.26,63.17,64.09,65.00,65.92,66.83,67.75,68.66,69.58,70.50,71.41,72.33,73.24,74.16,75.07,75.99,76.90,77.82,78.74,79.65,80.57,81.48,82.40,83.31,84.23,85.14,86.06,86.98,87.89,88.81,89.72,90.64,91.55,92.47,93.38,94.30,95.21,96.13,97.05,97.96,98.88,99.79,100.71,101.62,102.54,103.45,104.37,105.29,106.20,107.12,108.03,108.95,109.86,110.78,111.69,112.61,113.53,114.44,115.36,116.27,117.19,118.10,119.02,119.93,120.85,121.77,122.68,123.60,124.51,125.43,126.34,127.26,128.17,129.09,130.00,130.92,131.84,132.75,133.67,134.58,135.50,136.41,137.33,138.24,139.16,140.08,140.99,141.91,142.82,143.74,144.65,145.57,146.48,147.40,148.32,149.23,150.15,151.06,151.98,152.89,153.81,154.72,155.64,156.56,157.47,158.39,159.30,160.22,161.13,162.05,162.96,163.88,164.79,165.71,166.63,167.54,168.46,169.37,170.29,171.20,172.12,173.03,173.95,174.87,175.78,176.70,177.61,178.53,179.44,180.36,181.27,182.19,183.11,184.02,184.94,185.85,186.77,187.68,188.60,189.51,190.43,191.35,192.26,193.18,194.09,195.01,195.92,196.84,197.75,198.67,199.58,200.50,201.42,202.33,203.25,204.16,205.08,205.99,206.91,207.82,208.74,209.66,210.57,211.49,212.40,213.32,214.23,215.15,216.06,216.98,217.90,218.81,219.73] # same as variable 'frequencies' from PPG-derived PSD, range of 40-220 BPM with a resolution of 0.9137 BPM according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599
wind_smooth = 30 #beats 

# Plot soft-dtw barycenter averaging
def plot_helper(tb, barycenter, t, X):
	# plot all points of the data set
	for t, series in zip(t,X):
		plt.plot(t, series.ravel(), "k-", alpha=.2)
	# plot the given barycenter of them
	plt.plot(tb, barycenter.ravel(), "r-", linewidth=2)

if 'ACTES' in folderDSprior:
	prior_subs_info = pd.read_table(folderDSprior + 'subject-info.csv', sep = ',')
	priot_test_measure = pd.read_table(folderDSprior + 'test_measure.csv', sep = ',')
	str_prior = '_ACTES'
	subjects_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18]
else:
	str_prior = '_MAX-HIE'
	subjects_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# for subject_n in subjects_all: # uncomment this if you want to run the full MAX-HIE with a Leave-One-Subject-Out approach

# 	subject_analyzed = 'sub' + str(subject_n) # uncomment this if you want to run the full MAX-HIE with a Leave-One-Subject-Out approach

folderResults = "./HR_Priors/" + folderDS + subject_analyzed + '_' + session_id

os.makedirs(folderResults, exist_ok=True)

print("Subject " + subject_analyzed + ' Session ' + session_id)

if 'SUBMAX-HIE' in folderDS:
	subjects_prior = subjects_all # Full prior dataset (either MAX-HIE or ACTES) for SUBMAX-HIE
else:
	subjects_prior = [sub for sub in subjects_all if sub != int(subject_analyzed[3:])] # Leave-One-Subject-Out approach for MAX-HIE

# ===== STEP 1: Compute HR smoothed series and soft-dtw barycenter average ===== #
if not os.path.isfile(folderResults + "/t_hr_smoothed_series" + str_prior + ".mat"):
	hr_smoothed_norm_pop = []
	t_smoothed_pop = []
	for subject in subjects_prior:
		if 'ACTES' in folderDSprior:
			RR = np.array(priot_test_measure[priot_test_measure['ID'] == subject]['RR'].values)/1000
			HR = 60/RR
			t_RR = np.array(priot_test_measure[priot_test_measure['ID'] == subject]['time'].values)
			t_RR = t_RR + np.abs(t_RR[0])
		elif(os.path.isfile(folderDSprior + 'RR_sub' + str(subject) + '_visit1.mat')):
			mRR = loadmat(folderDSprior + 'RR_sub' + str(subject) + '_visit1.mat')
			t_RR = mRR['t_RR'].reshape(1, -1)[0].astype(float)
			RR = mRR['RR'].reshape(1, -1)[0].astype(float)
			HR = 60/RR
		# Smooth query HR series
		HR_series = pd.Series(HR).rolling(window=wind_smooth, center=True, min_periods=1).mean()
		HR_smoothed = HR_series.values
		t_smoothed = t_RR
		t_smoothed_pop.append(t_RR)
		# Normalize HR series for dtw barycenter averaging
		HR_smoothed_norm = HR_smoothed/np.max(HR_smoothed)
		hr_smoothed_norm_pop.append(HR_smoothed_norm)

	hr_smoothed_norm_pop_list = []
	t_smoothed_pop_list = []
	for t_series, hr_series in zip(t_smoothed_pop,hr_smoothed_norm_pop):
		t_smoothed_pop_list.extend(t_series.tolist())
		hr_smoothed_norm_pop_list.extend(hr_series.tolist())

	max_length = max(len(sublist) for sublist in t_smoothed_pop) # soft-dtw barycenter has the length of the longest series
	max_t = max(sublist[-1] for sublist in t_smoothed_pop) # final time of the longest series

	mHRsmoothed = {'t_smoothed_pop': t_smoothed_pop, 'hr_smoothed_norm_pop': hr_smoothed_norm_pop, 't_smoothed_pop_list': t_smoothed_pop_list, 'hr_smoothed_norm_pop_list': hr_smoothed_norm_pop_list}
	savemat(folderResults + "/t_hr_smoothed_series" + str_prior + ".mat",mHRsmoothed)
else:
	mHRsmoothed = loadmat(folderResults + "/t_hr_smoothed_series" + str_prior + ".mat")
	t_smoothed_pop = mHRsmoothed['t_smoothed_pop'][0].tolist()
	t_smoothed_pop = [t_s.T for t_s in t_smoothed_pop]
	hr_smoothed_norm_pop = mHRsmoothed['hr_smoothed_norm_pop'][0].tolist()
	hr_smoothed_norm_pop = [hr_s.T for hr_s in hr_smoothed_norm_pop]
	t_smoothed_pop_list = mHRsmoothed['t_smoothed_pop_list'][0]
	hr_smoothed_norm_pop_list = mHRsmoothed['hr_smoothed_norm_pop_list'][0]

	max_length = max(len(sublist) for sublist in t_smoothed_pop) # soft-dtw barycenter has the length of the longest series
	max_t = max(sublist[-1] for sublist in t_smoothed_pop)[0] # final time of the longest series
	
# Get HRmax from HR validation data for series normalization (RR intervals for MAX-HIE and subject B023 of SUBMAX-HIE, and Hexoskin HR for other subjects of SUBMAX-HIE)
if(os.path.isfile(folderHRval + 'RR_' + subject_analyzed + '_' + session_id + '.mat')):
	mRR = loadmat(folderHRval + 'RR_' + subject_analyzed + '_' + session_id + '.mat')
	t_RR = mRR['t_RR'].reshape(1, -1)[0].astype(float)
	RR = mRR['RR'].reshape(1, -1)[0].astype(float)
	HR = 60/RR
else:
	mHR = loadmat(folderHRval + 'hexoskin_HR_' + subject_analyzed + '_' + session_id + '.mat')
	t_RR = mHR['t'].reshape(1, -1)[0].astype(float)
	HR = mHR['y_raw'].reshape(1, -1)[0].astype(float)
# Smooth query HR series
HR_series = pd.Series(HR).rolling(window=wind_smooth, center=True, min_periods=1).mean()
HR_smoothed = HR_series.values
HRmax_sub = np.max(HR_smoothed)

# Use soft-dtw barycenter as reference for the dtw
if os.path.isfile(folderResults + "/softdtw_barycenter_reference" + str_prior + ".mat"):
	mSDTW = loadmat(folderResults + "/softdtw_barycenter_reference" + str_prior + ".mat")
	dtw_hr_reference = mSDTW['dtw_hr_reference'][0]
else:
	dtw_hr_reference = softdtw_barycenter(hr_smoothed_norm_pop, gamma=1., max_iter=50, tol=1e-3)
	dtw_hr_reference = [item[0] for item in dtw_hr_reference]
	dtw_hr_reference = np.array(dtw_hr_reference)
	mSDTW = {'dtw_hr_reference': dtw_hr_reference}
	savemat(folderResults + "/softdtw_barycenter_reference" + str_prior + ".mat",mSDTW)
dtw_t_reference = np.linspace(0,max_t,max_length) #soft-dtw barycenter has the length of the longest series

plt.figure(figsize=(16, 9))
plot_helper(dtw_t_reference, dtw_hr_reference, t_smoothed_pop, hr_smoothed_norm_pop)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Time (s)', fontsize=30)
plt.ylabel('Normalized HR', fontsize=30)
plt.title('Soft-DTW Barycenter Average of Prior Dataset')
if not os.path.isfile(folderResults + "/soft-dtw_averaging" + str_prior + ".png"):
	plt.savefig(folderResults + "/soft-dtw_averaging" + str_prior + ".png")
	plt.close()

# ===== STEP 2: Compute Probability Density Function (PDF) based on HR smoothed series ===== #

# 1. Multivariate kernel density estimator (KDE)
kde_t_hr = KDEMultivariate(data=[t_smoothed_pop_list,hr_smoothed_norm_pop_list],var_type='cc', bw='normal_reference')
t_grid = np.arange(0,int(max_t),1)
hr_grid = np.array(freqs)/HRmax_sub

# 2. Create the 2D grid
Time, HR = np.meshgrid(t_grid, hr_grid)
grid_points = np.vstack([Time.ravel(), HR.ravel()]).T

# 3. Evaluate the PDF on the grid and reshape back to 2D (original series)
if not os.path.isfile(folderResults + "/population-based_prior_hr_norm_t_every1s" + str_prior + ".mat"):
	pdf_values = kde_t_hr.pdf(grid_points)
	Z = pdf_values.reshape(Time.shape)
	mPDFhrECG = {'t_grid': t_grid, 'hr_grid': hr_grid, 'Time': Time, 'HR': HR, 'grid_points': grid_points, 'pdf_values': pdf_values, 'Z': Z}
	savemat(folderResults + "/population-based_prior_hr_norm_t_every1s" + str_prior + ".mat", mPDFhrECG)
else:
	mPDFhrECG = loadmat(folderResults + "/population-based_prior_hr_norm_t_every1s" + str_prior + ".mat")
	Z = mPDFhrECG['Z']

# 4. Plot PDF heatmap
plt.figure(figsize=(10, 6))
plt.pcolormesh(Time, HR, Z, shading='auto', cmap='turbo')
plt.colorbar(label='Probability Density')
plt.xlabel('Time (s)')
plt.ylabel('Normalized HR')
plt.title('Population-based Prior')
if not os.path.isfile(folderResults + "/population-based_prior_hr_norm_t_every1s" + str_prior + ".png"): 
	plt.savefig(folderResults + "/population-based_prior_hr_norm_t_every1s" + str_prior + ".png")	
	plt.close()

plt.show()