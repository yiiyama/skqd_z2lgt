import os
import sys
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp

from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.common import make_dual_lattice
from skqd_z2lgt.tasks.train_generator import train_for
from skqd_z2lgt.tasks.preprocess import load_reco

istep = int(sys.argv[1])
num_train = int(sys.argv[2])
gpu = sys.argv[3]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

with open('/data/iiyama/2dz2/ref_full156_20-24_1.5_0.4_1_kawasaki/parameters.json', 'r', encoding='utf-8') as src:
    parameters = Parameters.model_validate_json(src.read())

vtx_data, plaq_data = load_reco(parameters, etype='ref', idt=0, istep=istep)

rng = np.random.default_rng()

logger.info('Start training model for istep=%d with %d training samples', istep, num_train)

indices = np.arange(1_000_000)
rng.shuffle(indices)
idx_train = indices[:num_train]
idx_test = indices[num_train:num_train + 50_000]

train_u = vtx_data[idx_train]
train_v = plaq_data[idx_train]
test_u = vtx_data[idx_test]
test_v = plaq_data[idx_test]

best_model, record = train_for(train_u, train_v, test_u, test_v, parameters.crbm)

with h5py.File(f'/data/iiyama/2dz2/train_test/ik{istep}_n{num_train}.h5', 'w', libver='latest') as out:
    best_model.save(out)
    group = out.create_group('records')
    for key, data in record.items():
        group.create_dataset(key, data=data)
