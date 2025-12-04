import sys
from pathlib import Path
import numpy as np
import h5py
import jax
sys.path.append(str(Path(__file__).parents[1] / 'lib'))
from unitary_krylov import compute_gen_eigvals

jax.config.update('jax_enable_x64', True)

config = '''
   * *
  * * *
 * * * *
* * * * *
 * * * *
  * * *
'''
plaquette_energy = 1.
krylov_dim = 8
delta_ts = np.linspace(0.002, 0.102, 21)

gen_eigvals = compute_gen_eigvals(config, plaquette_energy, krylov_dim, delta_ts)
with h5py.File('/work/gp14/p14000/data/gen_eigvals_n27.h5', 'w', libver='latest') as out:
    out.create_dataset('gen_eigvals', data=gen_eigvals)
