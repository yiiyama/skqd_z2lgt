import os
import sys
import logging
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, AxisType

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)

num_plaq, max_nexc = map(int, sys.argv[1:3])
device = None
if len(sys.argv) == 4:
    gpus = list(map(int, sys.argv[3].split(',')))
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    if len(gpus) == 4:
        mesh = jax.make_mesh((4,), ('X',), (AxisType.Explicit,))
        device = NamedSharding(mesh, PartitionSpec(None, 'X'))

if device is None:
    device = jax.devices()[0]

out_dir = Path('/data/iiyama/2dz2/buildup')
out_file = h5py.File(out_dir / f'pt_states_{num_plaq}plaq.h5', 'a', libver='latest')

num_packed = np.ceil(num_plaq / 8).astype(int)
powers = jnp.power(256, jnp.arange(num_packed)).astype(np.uint64)[None, ::-1]
powers = jax.device_put(powers, device)

@jax.jit
def get_packed_states(states):
    states = states[:, None, :] | jnp.eye(num_plaq, dtype=np.uint8, device=device)[None, ...]
    states = states.reshape((-1, num_plaq))
    packed = jnp.sum(jnp.packbits(states, axis=1).astype(np.uint64) * powers, axis=1)
    packed = jnp.unique(packed, size=states.shape[0], fill_value=0)
    return packed

states = np.zeros((1, num_plaq), dtype=np.uint8)
if 'states_0' not in out_file:
    out_file.create_dataset('states_0', data=states)

for nexc in range(1, max_nexc + 1):
    print(nexc)
    name = f'states_{nexc}'
    if name in out_file:
        states = out_file[name][()]
        continue

    print('nexc', nexc)
    states = jax.device_put(states, device)
    print('states', states.shape, states.dtype)
    packed = get_packed_states(states)
    print('packed', packed.shape, packed.dtype)
    packed = np.array(packed[np.bitwise_count(packed) == nexc])
    print('uniquified', packed.shape, packed.dtype)
    states = np.unpackbits(((packed[:, None] // powers) % 256).astype(np.uint8), axis=1)
    states = states[:, :num_plaq]
    print('unpacked', states.shape, states.dtype)
    out_file.create_dataset(name, data=states)

out_file.close()
