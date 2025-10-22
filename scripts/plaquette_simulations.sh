#!/bin/bash

SCRIPTDIR=$(cd $(dirname $0); pwd)
OUTDIR=/data/iiyama/2dz2/plaqsim_data

for DT in 0.1 0.2 0.4 0.6
do
  sed 's/DELTAT/'$DT'/' $SCRIPTDIR/plaquette_simulation_conf.yaml > /tmp/conf.yaml
  python3 $SCRIPTDIR/run_plaquette_simulation.py \
      -o $OUTDIR/30plaqs_k0.7_8steps_dt${DT}_100kshots.h5 --log-level debug \
      /tmp/conf.yaml
done
