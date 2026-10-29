#===========================================================
# Builds symlink trees so archived WoFS and ProbSevere v3 data
# match the layout that wofs_phi / run_inference.py expect:
#
#   WoFS: <stage>/wofs/<date_before_00z>/<init>/wofs_ALL_*.nc  (inits merged across runs of one domain)
#     from <wofs_root>/WOFSRun<YYYYMMDD>-<HHMMSS>d<N>/<calendar date>/<init>/
#
#   PSv3: <stage>/ps/<YYYY>/<YYYYMMDD>/PROBSEVERE-V3/SSEC_AWIPS_PROBSEVERE-V3_<YYYYMMDD>_<HHMM>00.json
#     from <ps_root>/<YYYYMMDD>/PROBSEVERE-V3/SSEC_AWIPS_PROBSEVERE_<YYYYMMDD>_<HHMMSS>.json
#     Seconds are dropped, as in the training data. A file on an odd minute is moved to
#     the previous even minute if that slot is free (PS updates every 2 minutes).
#
# Then run run_inference.py with --wofs-dir <stage>/wofs --ps-dir <stage>/ps
#===========================================================

import argparse
import datetime as dt
import glob
import os
import re

WOFS_RUN_PATTERN = re.compile(r"^WOFSRun(\d{8})-(\d{6})d(\d+)$")
PS_PATTERN = re.compile(r"^SSEC_AWIPS_PROBSEVERE(?:-V3)?_(\d{8})_(\d{2})(\d{2})(\d{2})\.json$")
WOFS_RESET_HOUR = 12 #inits before this hour belong to the previous (before 00z) date


def find_wofs_runs(wofs_root, domain):
    '''Returns every WOFSRun directory for a domain. A case date's inits can be spread over
        several runs (restarts, runs started the previous evening, later reruns).'''

    return [os.path.join(wofs_root, run) for run in sorted(os.listdir(wofs_root))
            if (WOFS_RUN_PATTERN.match(run) is not None) and (WOFS_RUN_PATTERN.match(run).group(3) == domain)]


def stage_wofs(wofs_root, date, domain, stage_dir):
    '''Links each init of a case date from whichever run holds it. If several runs hold the
        same init, the copy with the most wofs_ALL files is used (earliest run on a tie).'''

    best = {} #init -> (num files, init dir)
    for run_dir in find_wofs_runs(wofs_root, domain):
        for cal_date in sorted(os.listdir(run_dir)):
            if not re.match(r"^\d{8}$", cal_date) or not os.path.isdir(os.path.join(run_dir, cal_date)):
                continue
            for init in sorted(os.listdir(os.path.join(run_dir, cal_date))):
                if not re.match(r"^\d{4}$", init):
                    continue
                init_dt = dt.datetime.strptime(cal_date + init, "%Y%m%d%H%M")
                if init_dt.hour < WOFS_RESET_HOUR:
                    init_dt -= dt.timedelta(days=1)
                if init_dt.strftime("%Y%m%d") != date:
                    continue
                init_dir = os.path.join(run_dir, cal_date, init)
                num_files = len(glob.glob(os.path.join(init_dir, "wofs_ALL_*.nc")))
                if num_files > 0 and ((init not in best) or (num_files > best[init][0])):
                    best[init] = (num_files, init_dir)

    if len(best) == 0:
        return "%s: no d%s WoFS inits" %(date, domain)

    out_date_dir = os.path.join(stage_dir, "wofs", date)
    os.makedirs(out_date_dir, exist_ok=True)
    for init, (__, init_dir) in best.items():
        link = os.path.join(out_date_dir, init)
        if os.path.islink(link) and os.readlink(link) != init_dir:
            os.remove(link)
        if not os.path.islink(link):
            os.symlink(init_dir, link)

    runs = sorted(set(os.path.basename(os.path.dirname(os.path.dirname(d))) for __, d in best.values()))
    return "%s: %s d%s WoFS inits from %s" %(date, len(best), domain, ", ".join(runs))


def stage_ps_day(ps_root, cal_date, stage_dir):
    '''Stages one calendar (UTC) day of PS files; returns (num staged, num skipped)'''

    files = sorted(glob.glob(os.path.join(ps_root, cal_date, "PROBSEVERE-V3", "*.json")))
    out_dir = os.path.join(stage_dir, "ps", cal_date[0:4], cal_date, "PROBSEVERE-V3")
    os.makedirs(out_dir, exist_ok=True)

    num_staged = 0
    num_skipped = 0
    for f in files:
        m = PS_PATTERN.match(os.path.basename(f))
        if m is None:
            num_skipped += 1
            continue
        file_date, hh, mm, __ = m.groups()
        minute = int(mm) - (int(mm) % 2)
        link = os.path.join(out_dir, "SSEC_AWIPS_PROBSEVERE-V3_%s_%s%02d00.json" %(file_date, hh, minute))
        if os.path.lexists(link):
            num_skipped += 1
            continue
        os.symlink(f, link)
        num_staged += 1

    return num_staged, num_skipped


def main():
    parser = argparse.ArgumentParser(description="Stage WoFS and PSv3 data for run_inference.py")
    parser.add_argument("--dates", nargs="+", required=True, help="case dates before 00z (YYYYMMDD)")
    parser.add_argument("--wofs-root", default="/work2/wof/realtime/SummaryFiles/2024")
    parser.add_argument("--ps-root", default="/work2/jcintineo/PSv3/2024")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--domain", default="1", help="WoFS domain number (the N in WOFSRun...dN)")
    args = parser.parse_args()

    for date in args.dates:
        print(stage_wofs(args.wofs_root, date, args.domain, args.stage_dir))

        #A case needs PS from 3 hours before the first init (1700Z) through the next day
        next_day = (dt.datetime.strptime(date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
        for cal_date in [date, next_day]:
            num_staged, num_skipped = stage_ps_day(args.ps_root, cal_date, args.stage_dir)
            print("%s: staged %s PS files (%s skipped: duplicate 2-min slot or already staged)" %(cal_date, num_staged, num_skipped))

    return


if (__name__ == '__main__'):

    main()
