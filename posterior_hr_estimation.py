# Estimating PPG-based HR series using population-based normalized HR distribution and selecting the best trajectory on the posterior
import os
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import *
import itertools as it
from trajectories import iter_trajectories
from sklearn.metrics import mean_absolute_error

# Analyzed subject settings and parameters
subject_analyzed = 'sub7' # for dataset MAX-HIE use 'subX'; for dataset SUBMAX-HIE use 'AXXX' or 'BXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"
session_id = 'visit1' # for dataset MAX-HIE use 'visit1'; for dataset SUBMAX-HIE use 'MTXXXX'; the subject and session IDs are reported in the file "PPG-Prior_Data/subject_session_ids.csv"
folderHRval = "./HR_Validation/MAX-HIE/" # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
folderDS = 'MAX-HIE/' # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
wind_smooth = 30 #beats

# PSD-PPG parameters
folderPSDPPG = './PSD_PPG/' + folderDS
wl = 8 # seconds 
overlap = wl-1 # seconds

# Prior parameters
folderPrior = './HR_Priors/' + folderDS + subject_analyzed + '_' + session_id + '/'
str_prior = '_MAX-HIE' # available priors: '_MAX-HIE', '_ACTES'

# Flags and parameters for trajectory selection with two different scoring metrics on the Posterior Probability Estimation (PPE)
flagTopTraj = 1 # 1: Top-Traj # 0: Min-DTW
if flagTopTraj == 1:
	str_scoring = '_Top-Traj' # top trajectory based on the cumulative log score over time on PPE
else:
	str_scoring = '_Min-DTW' # DTW alignment with soft-DTW barycenter average and selecting the trajectory with the minimum euclidean distance 
	start_dtw = 185 # seconds (start of exercise) # for selection based on Min-DTW. It can be applied on the full test (start_dtw = 0), but the results are overall worse

# Flags and parameters to plot normalized HR or real HR (BPM)
flagPlotNormHR = 1 # 1 if you want to plot the normalized HR
if flagPlotNormHR == 0: 
	str_norm = ''
else:
	str_norm = '_normHR'

folderResults = './Results/' + folderDS + subject_analyzed + '_' + session_id
os.makedirs(folderResults, exist_ok=True)

# Load smoothed HR series from prior population
mHRsmoothed = loadmat(folderPrior + 't_hr_smoothed_series' + str_prior + '.mat')
t_smoothed_pop = mHRsmoothed['t_smoothed_pop'][0].tolist()
t_smoothed_pop = [t_s.T for t_s in t_smoothed_pop]
hr_smoothed_norm_pop = mHRsmoothed['hr_smoothed_norm_pop'][0].tolist()
hr_smoothed_norm_pop = [hr_s.T for hr_s in hr_smoothed_norm_pop]
t_smoothed_pop_list = mHRsmoothed['t_smoothed_pop_list'][0]
hr_smoothed_norm_pop_list = mHRsmoothed['hr_smoothed_norm_pop_list'][0]

# Get HRmax from HR validation data for series normalization (RR intervals for MAX-HIE and subject B023 of SUBMAX-HIE, and Hexoskin HR for other subjects of SUBMAX-HIE)
if(os.path.isfile(folderHRval + 'RR_' + subject_analyzed + '_' + session_id + '.mat')):
	mRR = loadmat(folderHRval + 'RR_' + subject_analyzed + '_' + session_id + '.mat')
	t_RR = mRR['t_RR'].reshape(1, -1)[0].astype(float)
	RR = mRR['RR'].reshape(1, -1)[0].astype(float)
	HRecg = 60/RR
else:
	mHR = loadmat(folderHRval + 'hexoskin_heart_rate_' + subject_analyzed + '_' + session_id + '.mat')
	t_RR = mHR['t'].reshape(1, -1)[0].astype(float)
	HRecg = mHR['y_raw'].reshape(1, -1)[0].astype(float)
# Smooth query HR series
HR_series = pd.Series(HRecg).rolling(window=wind_smooth, center=True, min_periods=1).mean()
HR_smoothed = HR_series.values
HRmax_sub = np.max(HR_smoothed)

# Load Soft-DTW Barycenter as reference for Min-DTW scoring metric
if flagTopTraj == 0:
	mSDTW = loadmat(folderPrior + "softdtw_barycenter_reference" + str_prior + ".mat")
	dtw_hr_reference = mSDTW['dtw_hr_reference'][0]
	max_length = max(len(sublist) for sublist in t_smoothed_pop) # soft-dtw barycenter has the length of the longest series
	max_t = max(sublist[-1] for sublist in t_smoothed_pop) # final time of the longest series
	dtw_t_reference = np.linspace(0,max_t,max_length)

# === STEP 1: Posterior Probabilty Estimation (PPE) ===
# Build posterior probability estimation based 
# on power spectral density (PSD) of PPG (computed on 8s every 1s)
# and prior distribution of normalized smoothed HR series 
# from ECG (not considering the subject analyzed)

# Load PSD of PPG as HR-PPG observatios of HR-PPG
mPSDhrPPG = loadmat(folderPSDPPG + "time_freq_psd_" + subject_analyzed + "_" + session_id + ".mat")
time_windows = mPSDhrPPG['time_windows'][0]
frequencies = mPSDhrPPG['frequencies'][0]
PSDhrPPG = mPSDhrPPG['PSD']

# Load Probability Density Function (PDF) as HR-ECG prior
mPDFhrPrior = loadmat(folderPrior + "population-based_prior_hr_norm_t_every1s" + str_prior + ".mat")
PDFhrPrior = mPDFhrPrior['Z']
t_grid_prior = mPDFhrPrior['t_grid'][0]
hr_grid_prior = mPDFhrPrior['hr_grid'][0]

# Compute PPE as observations*prior
ind_t_end = min(t_grid_prior[-1],time_windows[-1])
PPE = PSDhrPPG[:,:ind_t_end-time_windows[0]+1] * PDFhrPrior[:,time_windows[0]:ind_t_end+1]
t_grid = t_grid_prior[time_windows[0]:ind_t_end+1]
hr_grid = hr_grid_prior

# === STEP 2: HR Trajectory Search and Selection ===

# 1. Trajectory generation (PPE normalized HR)
trajectories = []
t_step_traj = 1 # seconds
t_traj = t_grid[::t_step_traj]
epsilon = 1/65. # refers to normalized HR
local_eps = 1/100. # refers to normalized HR
tube_scale = 0.75 # default

# a. Collect a pool of candidate trajectories (e.g., first 50 candidates)
candidate_pool = list(
    it.islice(
        iter_trajectories(
            PPE[:, ::t_step_traj],
            hr_grid,
            epsilon,
            local_eps,
            tube_scale=tube_scale
        ),
        50  # Search up to 50 candidates to find the best 10
    )
)
# b. Sort the collected candidates by original log_score (index 1) in descending order
candidate_pool.sort(key=lambda x: x[1], reverse=True)

# c. Take the top 10 highest scoring trajectories
top_10_candidates = candidate_pool[:10] # This is done to reduce the computation of DTW alignment in case of Min-DTW scoring metric
for traj, _ in top_10_candidates: 
	trajectories.append(traj)

# 2. Select best trajectory (normalized HR)
if flagTopTraj == 1:
	traj_sel = trajectories[0]
else:
	dist_dtw_traj = []
	for traj,n_traj in zip(trajectories,range(10)):
		cut_traj = traj[np.where(t_traj>start_dtw)[0]] # only exercise part, cause the rest can change from subject to subject and it doesn't reflect the soft-dtw average
		alignment = dtw(cut_traj, dtw_hr_reference[np.where(dtw_t_reference>start_dtw)[0]], keep_internals=True, dist_method = 'seuclidean')
		traj_warped = cut_traj[alignment.index1]
		dist_dtw_traj.append(alignment.distance)
	traj_sel = trajectories[np.argmin(dist_dtw_traj)]
traj_sel = np.array(traj_sel).flatten()
t_traj = np.array(t_traj).flatten()

# === STEP3: Plot final graphs === #

# 1. Plot population-based prior
if flagPlotNormHR == 1:
	TimePrior, HRprior = np.meshgrid(t_grid_prior, hr_grid_prior) 
else:
	TimePrior, HRprior = np.meshgrid(t_grid_prior, hr_grid_prior*HRmax_sub) 
plt.figure(figsize=(16, 9))
pcolormesh = plt.pcolormesh(TimePrior, HRprior, PDFhrPrior, shading='auto', cmap='turbo')
cbar = plt.colorbar(pcolormesh)
cbar.set_label('Probability Density', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Time (s)', fontsize=30)
if flagPlotNormHR == 1:
	plt.ylabel('Normalized HR', fontsize=30)
	plt.title('Population-based Prior of Normalized HR Series', fontsize=20)	
else:
	plt.ylabel('HR (BPM)', fontsize=30)
	plt.title('Population-based Prior of HR Series', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=28)
if not os.path.isfile(folderResults + "/prior_distribution" + str_norm + str_prior + ".png"): 
	plt.savefig(folderResults + "/prior_distribution" + str_norm + str_prior + ".png",dpi=180)
	plt.close()

# 2. Plot PSD of PPG
plt.figure(figsize=(16, 9))
if flagPlotNormHR == 1:
	pcolor = plt.pcolor(time_windows, hr_grid, PSDhrPPG, cmap='turbo')
else:
	pcolor = plt.pcolor(time_windows, frequencies, PSDhrPPG, cmap='turbo')
cbar = plt.colorbar(pcolor)
cbar.set_label('Power Spectral Density', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Time (s)', fontsize=30)
if flagPlotNormHR == 1:
	plt.ylabel('Normalized HR', fontsize=30)
else:
	plt.ylabel('HR (BPM)', fontsize=30)
plt.title('Power Spectral Density Over Time - Subject ' + subject_analyzed + ' Session ' + session_id, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=28)
if not os.path.isfile(folderResults + "/psd_ppg" + str_norm + str_prior + ".png"): 
	plt.savefig(folderResults + "/psd_ppg" + str_norm + str_prior + ".png",dpi=180) 
	plt.close()

# 3. Plot heatmap of posterior
if flagPlotNormHR == 1: 
	TimePPE, HRppe = np.meshgrid(t_grid, hr_grid)
else:
	TimePPE, HRppe = np.meshgrid(t_grid, hr_grid*HRmax_sub)
plt.figure(figsize=(16, 9))
pcolormesh = plt.pcolormesh(TimePPE, HRppe, PPE, shading='auto', cmap='turbo')
cbar = plt.colorbar(pcolormesh)
cbar.set_label('Posterior Probability Estimation', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Time (s)', fontsize=30)
if flagPlotNormHR == 1:
	plt.ylabel('Normalized HR', fontsize=30)
else:
	plt.ylabel('HR (BPM)', fontsize=30)
plt.title('Posterior HR Estimation - Subject ' + subject_analyzed + " Session " + session_id, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=28)
if not os.path.isfile(folderResults + "/ppe" + str_norm + str_prior + ".png"): 
	plt.savefig(folderResults + "/ppe" + str_norm + str_prior + ".png") 
	plt.close()

# 4. Plot heatmap of posterior normalized with candidate trajectories
TimePPE, HRppe = np.meshgrid(t_grid, hr_grid)
plt.figure(figsize=(16, 9))
pcolormesh = plt.pcolormesh(TimePPE, HRppe, PPE, shading='auto', cmap='turbo')
cbar = plt.colorbar(pcolormesh)
cbar.set_label('Posterior Probability Estimation', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Time (s)', fontsize=30)
plt.ylabel('Normalized HR', fontsize=30)
plt.title('Posterior HR Estimation - Subject ' + subject_analyzed + " Session " + session_id, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=28)
for traj in trajectories:
	plt.plot(t_traj, traj, alpha=0.4, linewidth=3)
if not os.path.isfile(folderResults + "/ppe_trajectories_every" + str(t_step_traj) + "s" + str_norm + str_prior + ".png"): 
	plt.savefig(folderResults + "/ppe_trajectories_every" + str(t_step_traj) + "s" + str_norm + str_prior + ".png") 
	plt.close()

# 5. Plot final heatmap of posterior with selected trajectory
if flagPlotNormHR == 1: 
	Time, HR = np.meshgrid(t_grid, hr_grid)
else:
	Time, HR = np.meshgrid(t_grid, hr_grid*HRmax_sub)
plt.figure(figsize=(16, 9))
pcolormesh = plt.pcolormesh(Time, HR, PPE, shading='auto', cmap='turbo')
cbar = plt.colorbar(pcolormesh)
cbar.set_label('Posterior Probability Estimation', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.xlabel('Time (s)', fontsize=30)
if flagPlotNormHR == 1:
	plt.ylabel('Normalized HR', fontsize=30)
else:
	plt.ylabel('HR (BPM)', fontsize=30)
plt.title('Proposed PPG-exclusive Method on Posterior - Subject ' + subject_analyzed + ' Session ' + session_id, fontsize=20)
if flagPlotNormHR == 1:
	plt.plot(t_traj,traj_sel, color = 'lime', marker = 'v', markersize=10, linestyle='', label = 'HR-PPG proposed method')
else:
	plt.plot(t_traj,traj_sel*HRmax_sub, color = 'lime', marker = 'v', markersize=10, linestyle='', label = 'HR-PPG proposed method')
plt.tick_params(axis='both', which='major', labelsize=28)
    
# Comparison with HR-ECG ground-truth
if(os.path.isfile(folderHRval + 'RR_' + subject_analyzed + '_' + session_id + '.mat')):
	# Option 1: find mean HR ecg over wl seconds with overlap as the PSD of PPG
	meanHRecg = []
	t_meanHRecg = []
	for i_w in range(len(t_traj)):
		t_start = i_w * (wl - overlap)
		t_end = t_start + wl
		HRecg_wind = HRecg[np.where((t_RR>=t_start) & (t_RR<=t_end))]
		if len(HRecg_wind)>0:
			meanHRecg_wind = np.mean(HRecg_wind)
			meanHRecg.append(meanHRecg_wind)
			t_meanHRecg.append(t_RR[np.where(t_RR<=t_end)[0][-1]])
	series_HRecg_resampled = meanHRecg
	t_HRecg = t_traj
else:
	# Option 2: find mean HR ecg within t_traj (for HR series every 1s)
	series_HRecg_resampled = HRecg[np.where((t_RR>=t_traj[0]) & (t_RR<=t_traj[-1]))]
	t_HRecg = t_RR[np.where((t_RR>=t_traj[0]) & (t_RR<=t_traj[-1]))]
	t_end = np.min([t_traj[-1],t_HRecg[-1]])
	traj_sel = traj_sel[np.where(t_traj<=t_end)]
if flagPlotNormHR == 1:
	plt.plot(t_HRecg,series_HRecg_resampled/HRmax_sub, color = 'crimson', marker = 'o',markersize=10, linestyle='', label = 'HR-ECG')
else:
	plt.plot(t_HRecg,series_HRecg_resampled, color = 'crimson', marker = 'o',markersize=10, linestyle='', label = 'HR-ECG')

# Add BeliefPPG HR (PPG+ACC) if available
if(os.path.isfile(folderResults + "/beliefppg_estimated_hr.mat")):
	mBFPPG = loadmat(folderResults + "/beliefppg_estimated_hr.mat")
	t_hr_bfppg = mBFPPG['t_hr'][0]
	hr_bfppg = mBFPPG['hr_est'][0]
	ind_cut = np.where((t_hr_bfppg>=t_traj[0]) & (t_hr_bfppg<=t_traj[-1]))
	if flagPlotNormHR == 1:
		plt.plot(t_hr_bfppg[ind_cut],hr_bfppg[ind_cut]/HRmax_sub, color = 'orange', marker = 'd',markersize=10, linestyle='', label = 'HR BeliefPPG')
	else:
		plt.plot(t_hr_bfppg[ind_cut],hr_bfppg[ind_cut], color = 'orange', marker = 'd',markersize=10, linestyle='', label = 'HR BeliefPPG')

# Save comparison plot
plt.legend(fontsize=20)
if not os.path.isfile(folderResults + "/hr_series_selected" + str_norm + str_prior + str_scoring + ".png"): 
	plt.savefig(folderResults + "/hr_series_selected" + str_norm + str_prior + str_scoring + ".png",dpi=180) 

# Compute error metrics of HR-PPG
series_HRppg_sel = traj_sel*HRmax_sub
mae_HR = mean_absolute_error(series_HRecg_resampled,series_HRppg_sel)

print("HR-PPG vs HR-ECG")
print(f"Mean Absolute Error (MAE) in BPM: {np.round(mae_HR,2)}")

# Save .mat file with estimated HR results
data_to_save = {
	't_hr': t_traj,                   
	'hr_est': traj_sel * HRmax_sub,  
}

filename_mat = folderResults + "/estimated_hr" + str_prior + str_scoring + ".mat"
savemat(filename_mat, data_to_save)

plt.show()