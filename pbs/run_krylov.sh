#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=gp14
#PBS -l select=1
###PBS -l walltime=02:00:00

module load cuda/12.9

source /work/gp14/p14000/qii-miyabi-kawasaki/.venv_aarch64/bin/activate

for step in $(seq 6 8)
do
    python /work/gp14/p14000/skqd_z2lgt/krylov.py --config 4 7 --plaquette-energy 0.5 --idt 8 --step $step
done
for idt in $(seq 9 10)
do
    for step in $(seq 1 8)
    do
        python /work/gp14/p14000/skqd_z2lgt/krylov.py --config 4 7 --plaquette-energy 0.5 --idt $idt --step $step
    done
done
