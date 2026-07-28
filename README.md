# frdd-wofs-phi

WoFS-PHI provides spatial probabilities of tornadoes, wind, and hail within a given radius (e.g., 15 or 39km) and time window based on a blend of Warn-on-Forecast and ProbSevere data (as well as TORP data in a version of WoFS-PHI used for the Martz et al. paper discussed below). It is designed to provide useful guidance in the "watch-to-warning" time frame (lead times of about 4 hours or less). The product is best used to highlight preferred spatial corridors of future severe weather occurrence. Two WoFS-PHI versions are currently available within WoFS. "Forecast mode" (under the "ML Products Tab" on the WoFS Viewer) is updated once per WoFS initialization and predicts out to 4 hours of lead time. "Warning mode" (available as a contour overlay in the WoFS Viewer) updates every 5 minutes and always predicts for the next 1h or 2h. 

Real-time WoFS-PHI (including both "Forecast Mode" and "Warning Mode") considers only WoFS and ProbSevere and is currently run on the Microsoft Azure Cloud using the procedure outlined in the "Quick Start" section below. 

Instructions for how WoFS-PHI was trained for a recent research article, "The Impact of Target Dataset Choice on Machine Learning Prediction of Severe Hail, Wind, and Tornadoes" by Martz et al., are shown in the "WoFS-PHI Training for Martz et al. Paper: martz_latest branch" section. One distinguishing feature of this version of WoFS-PHI is it incorporates Tornado Probability Algorithm (TORP) data in addition to WoFS and ProbSevere. 



## Quick Start: Real-time WoFS-PHI setup and usage

### 1. Create the environment

```bash
conda env create -f environment_py310.yml
```

### 2. Activate the environment

```bash
conda activate frdd-wofs-phi
```

If your environment name is different, use that name instead.

### 3. Run a Python script with `MLGenerator`

```python
from wofs_phi.wofs_phi import MLGenerator

# all_files: collection of paths to wofs_ALL files for all previous timesteps
all_files = [
	"/path/to/all_timestep_1.nc",
	"/path/to/all_timestep_2.nc",
]

# prob_severe_files: collection of paths to ProbSevere obs files up to that point
prob_severe_files = [
	"/path/to/probsevere_obs_1.json",
	"/path/to/probsevere_obs_2.json",
]

WOFS_PROB_SVR_DIR = "/path/to/wofs_prob_svr"
tdir = "/path/to/output_or_working_dir"

ml = MLGenerator(all_files, prob_severe_files, WOFS_PROB_SVR_DIR, \
	tdir, [], tdir, 'forecast', ["obs_and_warnings"])
ml.generate()
```

Save this as a script (for example, `run_ml.py`) and run:

```bash
python run_ml.py
```
## WoFS-PHI Training for Martz et al. Paper: martz_latest branch

To generate data as in the Martz et al. paper, make sure you pull code from the martz_latest branch. 

Data availability: 
ProbSevere data is available at: https://registry.opendata.aws/noaa-mrms-pds/
Report data is available at: https://www.ncei.noaa.gov/stormevents/
Warning data is available at: https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml
WoFS and TORP data are available from NSSL upon request. 


### 1. Set the following variables in the config.py file as desired (other config variables can be left alone): 

include_torp_in_predictors

torp_vars_filename
raw_torp_training_dir
torp_dir

generate_forecasts #Generates the predictors array if True
generate_reports #Generates the reports file if True 
save_npy #Tells whether or not to save the npy predictor files 

#Change buffer time around reports
report_time_buffer = 20 

#Path to full_npy directory for training (for reports)
train_fcst_full_npy_dir
train_obs_full_npy_dir

#dat files are used for training for efficiency
train_fcst_dat_dir
train_obs_dat_dir

#Path to warnings directory
train_obs_and_warnings_full_2d_npy_dir
train_obs_and_warnings_full_1d_npy_dir
train_obs_and_warnings_sampled_dat_dir

train_warnings_full_2d_npy_dir
train_warnings_full_1d_npy_dir
train_warnings_sampled_dat_dir

train_warnings_csv_dir
train_warnings_full_2d_npy_dir
train_warnings_full_1d_npy_dir
train_warnings_sampled_1d_dat_dir

reps_coords_dir

wofs_base_path

#NOTE: These 4 should all be represent the same value
obs_radii
obs_radii_str
obs_radii_float
final_str_obs_radii

final_hazards

wofs_dir
ps_dir

### 2. Create the reports and predictor files for training. 

In gen_preds() of ml_driver, uncomment lines to calculate proceed_ps, proceed_torp, and proceed_wofs. Can optionally compute already_done variables to not repeat calculations. Key functions to call from gen_preds() is ml.generate(). This will create the reports and predictor files for training. The following variables should be set in the main() function of ml_driver():

report_radii
date_file
windows
lead_times

### 3. Generate warning label maps (if desired) 

Set the following variables at the top of main:

hazards
dates
starts
lengths
radii - only necessary if creating merged obs and warnings files
leads
model_types

Then, uncomment the sections (labeled by what each function does). You should have files for random sampling of warnings after generating report and/or feature files in step 2

### 4. Train the model 

Run wofs_phi.py with the following variables set in config.py (leave all other variables as you set them in step 1).

train_mode
train_types
train_hazards
train_lead_times
forecast_lengths
num_folds = 5