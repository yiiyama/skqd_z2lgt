import sys
import numpy as np
import h5py

in_name = sys.argv[1]
nout = int(sys.argv[2])
out_name = sys.argv[3]

with h5py.File(in_name, 'r', libver='latest') as source:
    probs = np.square(np.abs(source['eigvec'][()]))

indices = np.argsort(probs)[-nout:][::-1]
sorted_probs = probs[indices]

with h5py.File(out_name, 'w', libver='latest') as out:
    out.create_dataset('indices', data=indices)
    out.create_dataset('sorted_probs', data=sorted_probs)
