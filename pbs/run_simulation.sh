#!/bin/bash
#PBS -q debug-c
#PBS -W group_list=gp14
#PBS -l select=1:ompthreads=112
###PBS -l walltime=00:10:00

module load singularity

SIF=/work/gp14/p14000/singularity/qiskit-jax\:2.1.1_0.6.2.sif
SING_OPT='--bind /work/gp14/p14000'
SCRIPT=/home/p14000/qii-miyabi-kawasaki/scripts/run_simulation.py
CONF=/home/p14000/qii-miyabi-kawasaki/scripts/simulation_conf.yaml
OUT=/work/gp14/p14000/sim_out/${PBS_JOBID}.h5

export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS

cd $PBS_O_WORKDIR
singularity exec $SING_OPT $SIF python3 $SCRIPT $CONF -o $OUT
