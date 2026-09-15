#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=gp14
#PBS -o /work/gp14/p14000/job/monopole_area.out
#PBS -j oe
#PBS -l select=1
#PBS -l walltime=02:00:00

OUT=/work/gp14/p14000/data/monopole_area

cd /work/gp14/p14000
source qii-miyabi-kawasaki/.venv_aarch64/bin/activate
cd skqd_z2lgt/scripts
python monopole_area.py $OUT/diamondoid4.json 19 --out $OUT/ext --mu 2.725,4.6,20
