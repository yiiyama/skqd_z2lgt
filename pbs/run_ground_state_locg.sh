#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=gp14
#PBS -o /work/gp14/p14000/job/ground_4x8.out
#PBS -j oe
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=01:00:00

cd /work/gp14/p14000
source qii-miyabi-kawasaki/.venv_aarch64/bin/activate
cd skqd_z2lgt/scripts
mpirun python ground_state_locg.py 4 8 1.5 --gpus mpi --out /work/gp14/p14000/data
