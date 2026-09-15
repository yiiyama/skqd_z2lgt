#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=gr42
#PBS -o /work/gp14/p14000/job/ground_state_locg.out
#PBS -j oe
#PBS -l select=16:mpiprocs=1
#PBS -l walltime=01:00:00

cd /work/gp14/p14000
source qii-miyabi-kawasaki/.venv_aarch64/bin/activate
cd skqd_z2lgt/scripts
mpirun python ground_state_locg.py 3 11 0.1 --gpus mpi --out-dir /work/gp14/p14000/data --output probs
# mpirun python ground_state_locg.py 6 6 0.1 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs
# mpirun python ground_state_locg.py 6 6 0.8 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs
# mpirun python ground_state_locg.py 6 6 1.5 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs
# mpirun python ground_state_locg.py 6 6 2.2 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs
# mpirun python ground_state_locg.py 6 6 2.9 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs
# mpirun python ground_state_locg.py 6 6 20.0 --gpus mpi --out-dir /work/gp14/p14000/data --output weight_probs