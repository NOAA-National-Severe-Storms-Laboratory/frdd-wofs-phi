#!/bin/bash
#===========================================================
# Stages inputs and runs wofs-phi PSv3 + TORP inference for a
# list of case dates, one WoFS domain at a time.
#
#   ./run_all_cases.sh <dates file> <stage dir> <output root> [domains...]
#
# <dates file> has the case date (before 00z, YYYYMMDD) in its
# first column. Domain N output goes to <output root>/<date>/ for
# d1 and <output root>_dN/<date>/ otherwise. Dates with no inits
# for a domain are skipped. Finished cases are not rerun, so the
# script can be restarted after an interruption.
#
# Settings (environment): NPROCS (36), RF_JOBS (3),
# TORP_DIR, WOFS_ROOT, PS_ROOT.
#===========================================================

set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,16p' "$0"
    exit 1
fi

DATES_FILE=$1
STAGE=$2
OUT_ROOT=$3
shift 3
DOMAINS=${@:-1 2}

NPROCS=${NPROCS:-36}
RF_JOBS=${RF_JOBS:-3}
TORP_DIR=${TORP_DIR:-/work2/jcintineo/torp_csvs/radar}
WOFS_ROOT=${WOFS_ROOT:-/work2/wof/realtime/SummaryFiles/2024}
PS_ROOT=${PS_ROOT:-/work2/jcintineo/PSv3/2024}
ENV=/work/mflora/miniconda3/envs/wofs-tf

export PROJ_DATA=$ENV/share/proj
export PYTHONDONTWRITEBYTECODE=1 #the repo tracks .pyc files
PYTHON=$ENV/bin/python

cd "$(dirname "$0")"
DATES=$(awk '{print $1}' "$DATES_FILE")

for domain in $DOMAINS; do
    stage_d=$STAGE/d$domain
    out_d=$OUT_ROOT
    [ "$domain" != "1" ] && out_d=${OUT_ROOT}_d$domain

    echo "=== d$domain: staging into $stage_d"
    $PYTHON stage_inputs.py --dates $DATES --domain $domain --stage-dir $stage_d \
        --wofs-root $WOFS_ROOT --ps-root $PS_ROOT | grep -v "staged .* PS files"

    run_dates=$(for d in $DATES; do [ -d $stage_d/wofs/$d ] && echo $d; done || true)
    if [ -z "$run_dates" ]; then
        echo "=== d$domain: no dates to run"
        continue
    fi

    echo "=== d$domain: running $(echo $run_dates | wc -w) dates into $out_d"
    mkdir -p $out_d
    $PYTHON run_inference.py --dates $run_dates --wofs-dir $stage_d/wofs --ps-dir $stage_d/ps \
        --torp-dir $TORP_DIR --outdir $out_d --nprocs $NPROCS --rf-jobs $RF_JOBS
done
