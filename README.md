# PPG-Prior: Robust PPG-exclusive Heart Rate Estimation During Maximal Exercise Using Population-Based Priors
This repository contains the method presented in a paper for the 2026 Computing in Cardiology (CinC) conference, titled "Robust PPG-exclusive Heart Rate Estimation During Maximal Exercise Using Population-Based Priors". The method has been validated on the maximal (**MAX-HIE**) and submaximal (**SUBMAX-HIE**) graded cycloergometer exercise tests provided in *PPG-Prior_Data/*. The ground-truth HR data derived from ECG used for validation are in *HR_Validation/*, divided by dataset. The one in **MAX-HIE** are also used to build the priors, with a Leave-One-Subject-Out approach for itself and fully for **SUBMAX-HIE**. 

To apply our ```PPG-Prior``` method, there are 3 main files:

- ```psd_ppg.py```: This script computes the Power Spectral Density (PSD) over time of the PPG signals provided in *PPG-Prior_Data/*. 
- ```population_based_hr_priors.py```: This script builds the HR dynamics prior over a population of subjects different from the one analyzed, in the form of Probability Density Function (PDF). The default prior dataset is **MAX-HIE**. However, for the data in **SUBMAX-HIE**, we have also tested the RR intervals provided in the Physionet [**ACTES**](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/) dataset containing cardiorespiratory measurement from graded cycloergometer exercise testing. You would need to download the dataset and save it in a folder *HR_Validation/ACTES/*.
- ```posterior_hr_estimation.py```: This script combines the PSD and the PDF to compute a Posterior Probability Estimation (PPE), as PPE = PSD × PDF. Then, the algorithm searches for candidate HR trajectories and selects the best one with the option of two different scoring methods. This search is done in the script ```trajectories.py``` imported in the script.

```trajectories.py``` is a Viterbi-based search algorithm and it contains few parameters that are set according to physiological changes in the HR. However, the most relevant parameter is ```local_eps```, which represents the maximum absolute HR change between consecutive time windows. As the algorithm is designed to work with normalized HR, ```local_eps``` also refers to normalized HR. This parameter can be changed in ```posterior_hr_estimation.py``` as input to the function call. 

The example subject provided in the scripts is *sub7* of the dataset **MAX-HIE**. There are clear instructions in the comments to change the ```subject_id```, ```session_id```, and ```folderDS```, if you want to run the algorithm for other subjects of **MAX-HIE** or **SUBMAX-HIE**.

Additionally, we provide a script ```evaluate_beliefppg.py``` to apply the state-of-the-art [```BeliefPPG```](https://github.com/eth-siplab/BeliefPPG) algorithm on the data in *BeliefPPG_Data/* divided into the same **MAX-HIE** and **SUBMAX-HIE** datasets. The folder contains the data in ```.npy ``` format including the same PPG and ground truth HR provided in *PPG-Prior_Data/* and *HR_Validation/*, and, additionally, as required by ```BeliefPPG```, the 3-axis accelerometer (ACC) data collected during the same maximal and submaximal tests, on the same position as the PPG.

For reproducibility, we report here the table of results for the full **MAX-HIE** and **SUBMAX-HIE** dataset both for ```BeliefPPG``` and ```PPG-Prior```, with the latter using the two scoring methods, defined *Top-Traj* and *Min-DTW* in the paper. For **SUBMAX-HIE**, we also report the results using the Physionet **ACTES** prior. The metric reported is the Mean Absolute Error (MAE) in BPM of each algorithm relative to the ground-truth ECG. In the paper, we report the percentage MAE drop of our ```PPG-Prior``` method compared to ```BeliefPPG``` as 100*(1 - MAE_PPG-Prior/MAE_BeliefPPG). **Note**: ```BeliefPPG``` could have small differences in BPM if run multiple times on the same subject due to the nature of the algorithm; we only observed a variation of up to approximately 0.5 BPM. 

## MAX-HIE Per-subject results
 
| Subject | BeliefPPG (PPG+ACC) | BeliefPPG (Only PPG) | PPG-Prior (Top-Traj) | PPG-Prior (Min-DTW) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| sub1  | 60.46 | 67.32 | 23.42 | 24.60 |
| sub2  | 24.22 | 68.83 | 8.88  | 9.07  | 
| sub3  | 13.52 | 22.99 | 3.86  | 3.08  |
| sub4  | 28.11 | 60.12 | 4.12  | 6.18  |
| sub5  | 36.74 | 56.79 | 20.09 | 14.55 |
| sub6  | 38.69 | 47.49 | 15.91 | 18.00 |
| sub7  | 16.10 | 32.10 | 2.92  | 3.69  |
| sub8  | 21.78 | 32.74 | 9.95  | 12.35 |
| sub9  | 15.61 | 52.99 | 10.98 | 13.99 |
| sub10 | 46.85 | 66.47 | 13.40 | 10.94 |
| sub11 | 21.35 | 29.67 | 20.00 | 15.11 |
| sub13 | 4.54  | 20.76 | 2.76  | 3.51  |
| sub14 | 12.96 | 40.01 | 2.54  | 3.08  |
| sub15 | 22.35 | 63.17 | 6.65  | 6.65  |
| sub16 | 4.53  | 23.15 | 2.38  | 3.40  |
| sub17 | 15.97 | 52.91 | 2.33  | 2.33  |
| sub18 | 27.96 | 42.23 | 9.83  | 7.25  |
| sub19 | 21.11 | 56.73 | 10.69 | 8.41  |
| sub20 | 28.18 | 56.90 | 7.14  | 6.56  |
| sub21 | 4.62  | 33.65 | 1.74  | 3.67  |
| sub22 | 19.23 | 37.65 | 11.94 | 15.95 |

## SUBMAX-HIE Per-subject results
| Subject | BeliefPPG (PPG+ACC) | BeliefPPG (Only PPG) | PPG-Prior (MAX-HIE, Top-Traj) | PPG-Prior (ACTES, Top-Traj) | PPG-Prior (MAX-HIE, Min-DTW) | PPG-Prior (ACTES, Min-DTW) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| A000  | 3.60  | 3.44  | 5.71  | 2.68 | 5.30  | 4.44  |
| A001  | 4.04  | 4.03  | 3.71  | 3.13 | 6.81  | 14.06 | 
| A002  | 15.56 | 47.41 | 39.23 | 1.38 | 30.76 | 18.06 |
| B001  | 3.75  | 3.78  | 1.98  | 1.63 | 1.98  | 16.98 |
| B023  | 5.64  | 5.45  | 2.86  | 2.87 | 3.73  | 2.87  |

## 

Please, cite the corresponding paper when using the code (DOI and BIBTEX citation COMING SOON).

For any question or additional information, please contact me through my [Linkedin](https://www.linkedin.com/in/elisabetta-de-giovanni-a3a804a9/) profile. 