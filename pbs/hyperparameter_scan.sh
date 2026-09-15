#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=gr42
#PBS -o /work/gp14/p14000/job/hyperparameter_scan.out
#PBS -j oe
#PBS -l select=1
#PBS -l walltime=01:00:00

DATADIR=/work/gp14/p14000/data/hyperparameter_scan_${PBS_ARRAY_INDEX}
DATASOURCE=/work/gp14/p14000/data/experiment_pittsburgh_may22
SCRIPT=/work/gp14/p14000/skqd_z2lgt/scripts/hyperparam_search.py
LOGPATH=/work/gp14/p14000/job/hyperparameter_scan

[ -d $DATADIR ] && rm -rf $DATADIR
mkdir $DATADIR
ln -s $DATASOURCE/data $DATADIR/data
cp $DATASOURCE/parameters.json $DATADIR/parameters.json

source /work/gp14/p14000/qii-miyabi-kawasaki/.venv_aarch64/bin/activate

python $SCRIPT $DATADIR $PBS_ARRAY_INDEX > $LOGPATH/${PBS_ARRAY_INDEX}.out 2>&1
