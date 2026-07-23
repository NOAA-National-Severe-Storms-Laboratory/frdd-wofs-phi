# frdd-wofs-phi

Official repository for the WoFS PHI project.

## Quick Start

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
