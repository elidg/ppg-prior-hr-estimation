# Script to evaluate power spectral density of PPG signal for HR estimation

# Load useful packages
import os, math
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, periodogram, detrend
from scipy.fftpack import fftfreq, rfft, irfft
import numpy as np
import matplotlib.pyplot as plt

# SETTINGS
folderData = './PPG-Prior_Data/'
folderDB = 'SUBMAX-HIE/' # available datasets: 'MAX-HIE' and 'SUBMAX-HIE'
subject = 'B001' # for dataset MAX-HIE use 'subX'; for dataset SUBMAX-HIE use 'AXXX' or 'BXXX'; the subject and session IDs are reported in the file "subject_session_ids.csv"
session_id = 'MT0002' # for dataset MAX-HIE use 'visit1'; for dataset SUBMAX-HIE use 'MTXXXX'; the subject and session IDs are reported in the file "subject_session_ids.csv"
folderResults = './PSD_PPG/'

# Signal parameters
fs = 176 # Hz, change this accordingly (250 Hz for MAX-HIE, 176 Hz for SUBMAX-HIE)
lowcut = 0.5 # Hz, *60 = min HR according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599
highcut = 10.0 # Hz, *60 = max HR according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599

# PSD parameters
res_fft = 0.0153 # Hz, approximate, according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599
NFFT = int(np.round(64*fs*1.024)) # 64s according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599# int(np.round(fs/res_fft))
low_bpm = 40 # from C code, 40 BPM according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599
high_bpm = 220 # from C code, 220 BPM according to De Giovanni et al. https://ieeexplore.ieee.org/document/7723599

# Segmentation parameters
wl = 8 # seconds
overlap = wl-1 # seconds

## === MAIN FUNCTIONS === ## 
# Filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Segmentation
def segment_time_vector(t, window_length, overlap):
	# Calculate the number of windows
	t_length = t[-1];
	t_start = 0;

	num_windows = math.ceil((t_length - window_length + 1) / (window_length - overlap));

	# Initialize the output array
	windows = [];
	indices = [];
    
	# Calculate the start and end indices for each window
	for i in range(num_windows):
		idx_wind = np.where((t >= t_start + i*(window_length - overlap)) & (t <= t_start + (i+1)*(window_length) - i*overlap))[0];

		if len(idx_wind)>0: 
			# Store the window indices
			windows.append(t[idx_wind])
			indices.append(idx_wind)

	if len(indices)>0:
		last_window = indices[-1]
		last_idx_wind = np.where(t>=t[last_window[-1]]-overlap)[0]

		if len(last_idx_wind)>0:
			windows.append(t[last_idx_wind])
			indices.append(last_idx_wind)

	return windows, indices

def bandpass_filter_rfft(t,sig):

	# Discrete FFT for real input
	W = fftfreq(sig.size, d=t[1]-t[0])
	f_signal = rfft(sig)
	# If our original signal time was in seconds, this is now in Hz    
	cut_f_signal = f_signal.copy()
	cut_f_signal[(W>highcut)] = 0
	cut_f_signal[(W<lowcut)] = 0
	filt_signal = irfft(cut_f_signal)

	return filt_signal

## === MAIN SCRIPT === ##

# Get PPG data
m = loadmat(folderData + folderDB + 'ppg_' + subject + '_' + session_id + '.mat')
t = m['t'].reshape(1, -1)[0].astype(float)
ppg = m['ppg_raw'].reshape(1, -1)[0].astype(float)

# Segment time vector
t_windows, ind_windows = segment_time_vector(t,wl,overlap)

# ===== STEP 1: Preprocessing =====
# 1 - 1st order Butterworth bandpass filter, cutoff freqs [0.5, 10] Hz
ppg_cleaned = butter_bandpass_filter(ppg, lowcut, highcut, fs, order=1)

# 2 - Detrend PPG cleaned signal
ppg_cleaned_detrended = detrend(ppg_cleaned)

# ===== STEP 2: Compute PSD window by window =====
wl_new = int(wl*fs)
wl_old = int(wl*fs)
overlap_sample = int(overlap*fs)
wind_start = 0
wind_end = wl_new
i_wind = 0
PSD = []
for sample in range(len(ppg_cleaned_detrended)):
	if(sample == wind_end or sample == len(ppg_cleaned_detrended)-1):
		ind_wind = np.arange(wind_start,wind_end)
		t_wind = t[ind_wind]
		ppg_window = ppg_cleaned_detrended[ind_wind]
		ppg_window_filt = bandpass_filter_rfft(t_wind,ppg_window)
		freqs_wind,psd_wind = periodogram(ppg_window_filt, fs, nfft = NFFT)
		norm_psd_wind = psd_wind / np.max(psd_wind)
		freqs_wind_cut = freqs_wind[np.where((freqs_wind>=low_bpm/60) & (freqs_wind<high_bpm/60))]
		freqs_wind_cut = np.round(freqs_wind_cut*60,decimals=2) # bpm
		if i_wind == 0:
			frequencies = freqs_wind_cut
		norm_psd_wind_cut = norm_psd_wind[np.where((freqs_wind>=low_bpm/60) & (freqs_wind<high_bpm/60))]
		PSD.append(norm_psd_wind_cut)

		wind_start = sample-overlap_sample
		wind_end = sample+wl_new-overlap_sample
		if wind_end > len(ppg_cleaned_detrended):
			wind_end = len(ppg_cleaned_detrended)
		i_wind+=1

PSD = np.array(PSD)
PSD = PSD.T
time_windows = np.arange(0,PSD.shape[1])+wl

mPSDPPG = {'time_windows': time_windows, 'frequencies': frequencies, 'PSD': PSD}
os.makedirs(folderResults + folderDB, exist_ok=True) 
savemat(folderResults + folderDB + "time_freq_psd_" + subject + "_" + session_id + ".mat", mPSDPPG)

# ===== STEP 3: Plot Heatmap of PPG power spectral density over time =====
plt.figure(figsize=(10, 6))
plt.pcolor(time_windows, frequencies, PSD, cmap='turbo')
plt.xlabel('Time Window Index')
plt.colorbar(label='PSD Value')
plt.ylabel('Frequency (BPM)')
plt.title('Power Spectral Density Over Time - Subject ' + subject + ' Session ' + session_id)

if not os.path.isfile(folderResults + folderDB + "heatmap_ppg_psd_" + subject + "_" + session_id + ".png"):
    plt.savefig(folderResults + folderDB + "heatmap_ppg_psd_" + subject + "_" + session_id + ".png")

plt.show()
plt.close()