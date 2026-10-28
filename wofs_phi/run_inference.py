#===========================================================
# Runs wofs-phi inference on archived cases with the
# WoFS + ProbSevere v3 + TORP, obs-trained, 5-fold RF models
# (Martz 2025: wofs_psv3_with_torp). Writes the raw RF
# probabilities (mean over folds) to netcdf -- no SR mapping.
#
# Run from this directory (see --help), e.g.:
#   python run_inference.py --dates 20240507 \
#       --wofs-dir /path/to/SummaryFiles --ps-dir /path/to/PS_json \
#       --torp-dir /path/to/torp --outdir /path/to/out
#===========================================================

import argparse
import datetime as dt
import os
import re
import sys
import time
import traceback
import warnings
from multiprocessing import Pool

import numpy as np
import xarray as xr

#Import the local config first so wofs_phi/ml_driver pick up this copy
import config as c
import wofs_phi
from ml_driver import MLDriver, WofsFile

MODEL_ROOT = "/work/ryan.martz/wofs_phi_data/obs_train/models/wofs_psv3_with_torp"
TRAIN_TYPE = "obs"
WINDOW = 60 #forecast window (minutes); only 60-min models were trained
LEAD_TIMES = [30, 60, 90, 120, 150, 180] #wofs init to start of valid period (minutes)
HAZARD_RADII = {"hail": "39", "wind": "375", "tornado": "39"} #obs radius (km) in model names
NUM_FOLDS = 5
WOFS_30MIN_INIT_LENGTH = 180 #bottom-of-the-hour wofs inits only run 3 hours

TORP_FNAME_PATTERN = re.compile(r"^(\d{8}-\d{6})_.*\.csv$")

_model_cache = {}
_torp_index_cache = {}


def set_config(args):
    '''Overrides the training-oriented settings in config.py for inference'''

    assert c.model_type == "wofs_psv3_with_torp" and c.num_training_vars == 374, \
        "config.py must be set to ps_version=3, include_torp_in_predictors=True, radar_data=True, filtered_torp=False"

    c.generate_forecasts = True
    c.generate_reports = False
    c.save_npy = False
    c.save_ncdf = False
    c.plot_forecasts = False
    c.is_train_mode = False #training-only TORP 00z linking relies on training directory layout
    c.use_ALL_files = True
    c.final_hazards = ["hail", "wind", "tornado"] #PS predictors are built for each of these
    c.wofs_base_path = args.wofs_dir
    c.ps_dir = args.ps_dir
    c.torp_dir = args.torp_dir

    return


def get_model_filenames(hazard, lead_time, folds):
    radius = HAZARD_RADII[hazard]
    model_dir = "%s/%s/wofslag_%s/length_%s" %(MODEL_ROOT, hazard, c.wofs_spinup_time, WINDOW)
    return ["%s/%s_%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm_fold%s.pkl" %(model_dir, c.model_type,
            TRAIN_TYPE, hazard, WINDOW, lead_time, lead_time + WINDOW, radius, fold) for fold in folds]


def load_models(hazard, lead_time, folds, rf_jobs):
    '''Loads (and caches) the fold models for one hazard/lead time. Only one lead time
        is kept in memory at a time.'''

    key = (hazard, lead_time)
    if key not in _model_cache:
        for k in [k for k in _model_cache if k[1] != lead_time]:
            del _model_cache[k]
        models = []
        for fname in get_model_filenames(hazard, lead_time, folds):
            model = wofs_phi.MLPrediction.load_rf_model(fname)
            model.n_jobs = rf_jobs
            models.append(model)
        _model_cache[key] = models

    return _model_cache[key]


def get_torp_index(torp_dir, date_before_00z):
    '''Returns a sorted list of (datetime, path) for every TORP csv under torp_dir whose
        filename starts with YYYYMMDD-HHMMSS_, restricted to the case date and the next day.
        If the same filename is found more than once (e.g., raw and QC/ copies), the copy
        in a QC directory is used.'''

    if date_before_00z not in _torp_index_cache:
        next_day = (dt.datetime.strptime(date_before_00z, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
        paths = {}
        for root, __, files in os.walk(torp_dir):
            for fname in files:
                m = TORP_FNAME_PATTERN.match(fname)
                if m and m.group(1)[0:8] in (date_before_00z, next_day):
                    if (fname not in paths) or (os.path.basename(root) == "QC"):
                        paths[fname] = os.path.join(root, fname)
        index = [(dt.datetime.strptime(fname[0:15], "%Y%m%d-%H%M%S"), path) for fname, path in paths.items()]
        index.sort()
        _torp_index_cache[date_before_00z] = index

    return _torp_index_cache[date_before_00z]


def find_torp_files(torp_dir, date_before_00z, wofs_init_dt):
    '''TORP files from the last c.torp_file_search_minutes before wofs init + spinup
        (same window as TORP.get_torp_files)'''

    most_recent = wofs_init_dt + dt.timedelta(minutes=c.wofs_spinup_time)
    earliest = most_recent - dt.timedelta(minutes=c.torp_file_search_minutes)

    return [path for t, path in get_torp_index(torp_dir, date_before_00z) if earliest <= t <= most_recent]


def get_cases(dates, inits, lead_times):
    '''Returns the list of (date, init, lead_time) cases, ordered by lead time so the
        model cache gets reused.'''

    cases = []
    for lead_time in lead_times:
        for date in dates:
            for init in inits:
                if (init in c.bottom_hour_inits) and (lead_time + WINDOW > WOFS_30MIN_INIT_LENGTH):
                    continue
                cases.append((date, init, lead_time))

    return cases


def check_inputs(wofs_path, wofs_files, ps_files):
    '''Returns an error string if inputs are missing, else "".
        Missing WoFS/PS files are otherwise skipped silently by wofs_phi.'''

    missing_wofs = [f for f in wofs_files if not os.path.isfile("%s/%s" %(wofs_path, f))]
    if len(missing_wofs) > 0:
        return "missing %s/%s WoFS files (e.g., %s/%s)" %(len(missing_wofs), len(wofs_files), wofs_path, missing_wofs[0])

    #Like training: need a PS file within c.ps_recent_file_threshold minutes of the PS init time
    num_recent = int(c.ps_recent_file_threshold/c.ps_update_rate) + 1
    if not any(os.path.isfile("%s/%s" %(c.ps_dir, f)) for f in ps_files[0:num_recent]):
        return "no PS file within %s min of %s/%s" %(c.ps_recent_file_threshold, c.ps_dir, ps_files[0])

    return ""


def save_ncdf(outdir, fcst_specs, grid, probs_dict, model_files_dict, num_torp_files):
    nc_outname = "%s/%s" %(outdir, wofs_phi.MLGenerator.get_ncdf_outname(fcst_specs))

    ds = xr.Dataset(coords={"y": range(grid.ny), "x": range(grid.nx)})
    for hazard, probs in probs_dict.items():
        var_name = "wofsphi__%s__%skm__%smin__%s" %(hazard, HAZARD_RADII[hazard], WINDOW, TRAIN_TYPE)
        ds[var_name] = (["y", "x"], probs.reshape(grid.ny, grid.nx).astype(np.float32))
        ds[var_name].attrs = {"description": "raw RF probability (no SR mapping), mean over folds",
                              "model_files": ", ".join(os.path.basename(f) for f in model_files_dict[hazard])}
    ds["xlat"] = (["y", "x"], np.asarray(grid.lats, dtype=np.float32))
    ds["xlon"] = (["y", "x"], np.asarray(grid.lons, dtype=np.float32))
    ds.attrs = {"model_type": c.model_type, "train_type": TRAIN_TYPE,
                "wofs_init_time": fcst_specs.wofs_init_time_dt.strftime("%Y%m%d_%H%M"),
                "ps_init_time": fcst_specs.ps_init_time_dt.strftime("%Y%m%d_%H%M"),
                "start_valid": fcst_specs.start_valid_dt.strftime("%Y%m%d_%H%M"),
                "end_valid": fcst_specs.end_valid_dt.strftime("%Y%m%d_%H%M"),
                "num_torp_files": num_torp_files}
    ds.to_netcdf(nc_outname)

    return nc_outname


def run_case(case_args):
    (date, init, lead_time), args = case_args
    case_str = "%s %s lead%03d" %(date, init, lead_time)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    try:
        wofs_path = MLDriver.find_wofs_path(date, init)
        wofs_files = MLDriver.find_wofs_file_list(WINDOW, init, lead_time, date)
        ps_files = MLDriver.find_ps_files_from_first_wofs(wofs_files[0], "forecast")

        error = check_inputs(wofs_path, wofs_files, ps_files)
        if error != "":
            return "SKIP %s: %s" %(case_str, error)

        wofs_init_dt = WofsFile.get_wofs_dt(init, date)
        torp_files = find_torp_files(args.torp_dir, date, wofs_init_dt)

        ml = wofs_phi.MLGenerator(wofs_files, ps_files, c.ps_dir, wofs_path, torp_files, args.outdir,
                                  "forecast", [TRAIN_TYPE])
        grid = wofs_phi.Grid.create_wofs_grid(wofs_path, wofs_files[0])
        fcst_specs = wofs_phi.ForecastSpecs.create_forecast_specs(ps_files, wofs_files, c.all_fields_file,
                                                                  c.all_methods_file, c.single_pt_file)
        __, predictors = ml.build_predictors(grid, fcst_specs)

        if predictors.shape[1] != c.num_training_vars:
            return "FAIL %s: built %s predictors, models expect %s" %(case_str, predictors.shape[1], c.num_training_vars)

        if args.save_predictors:
            np.save("%s/predictors_%s_%s_lead%03d.npy" %(args.outdir, date, init, lead_time), predictors.astype(np.float32))

        probs_dict = {}
        model_files_dict = {}
        for hazard in args.hazards:
            models = load_models(hazard, lead_time, args.folds, args.rf_jobs)
            probs_dict[hazard] = np.mean([m.predict_proba(predictors)[:, 1] for m in models], axis=0)
            model_files_dict[hazard] = get_model_filenames(hazard, lead_time, args.folds)

        nc_outname = save_ncdf(args.outdir, fcst_specs, grid, probs_dict, model_files_dict, len(torp_files))

        return "OK   %s: %s (%s TORP files, %.0fs)" %(case_str, nc_outname, len(torp_files), time.time() - t0)

    except Exception:
        return "FAIL %s:\n%s" %(case_str, traceback.format_exc())


def main():
    all_inits = c.top_hour_inits + c.bottom_hour_inits

    parser = argparse.ArgumentParser(description="wofs-phi inference with the WoFS + PSv3 + TORP obs-trained fold models")
    parser.add_argument("--dates", nargs="+", required=True, help="case dates before 00z (YYYYMMDD)")
    parser.add_argument("--wofs-dir", required=True, help="root dir holding <date>/<init>/wofs_ALL_*.nc")
    parser.add_argument("--ps-dir", required=True,
                        help="root dir holding YYYY/YYYYMMDD/PROBSEVERE-V3/SSEC_AWIPS_PROBSEVERE-V3_YYYYMMDD_HHMM00.json")
    parser.add_argument("--torp-dir", required=True,
                        help="root dir searched recursively for TORP csvs named YYYYMMDD-HHMMSS_*.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--inits", nargs="+", default=all_inits, help="wofs init times (HHMM); default: all")
    parser.add_argument("--leads", nargs="+", type=int, default=LEAD_TIMES, choices=LEAD_TIMES)
    parser.add_argument("--hazards", nargs="+", default=list(HAZARD_RADII), choices=list(HAZARD_RADII))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(NUM_FOLDS)),
                        help="fold models to average (2024 is out of sample for all folds)")
    parser.add_argument("--nprocs", type=int, default=1, help="cases to run in parallel")
    parser.add_argument("--rf-jobs", type=int, default=4, help="n_jobs for each RF predict_proba")
    parser.add_argument("--save-predictors", action="store_true", help="also save the 1d predictor arrays (.npy)")
    args = parser.parse_args()

    set_config(args)
    os.makedirs(args.outdir, exist_ok=True)

    cases = get_cases(args.dates, args.inits, args.leads)
    print("%s cases; model_type=%s; %s predictors; folds=%s" %(len(cases), c.model_type, c.num_training_vars, args.folds))
    sys.stdout.flush()

    case_args = [(case, args) for case in cases]
    if args.nprocs > 1:
        with Pool(args.nprocs) as pool:
            for result in pool.imap(run_case, case_args):
                print(result)
                sys.stdout.flush()
    else:
        for ca in case_args:
            print(run_case(ca))
            sys.stdout.flush()

    return


if (__name__ == '__main__'):

    main()
