"""Run a hyperparameter search."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, train_crbm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.WARNING)
logging.getLogger('skqd_z2lgt').setLevel(logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Callback(DefaultCallback):
    """Callback that calculates nll in addition to loss and free energy."""
    def __init__(self, eval_every=10):
        super().__init__(eval_every=eval_every)
        self.metrics._metric_names.append('nll')
        setattr(self.metrics, 'nll', nnx.metrics.Average('nll'))

    @nnx.jit
    def _train_step_ext(self, model, u_batch, v_batch, updates):
        """Callback within train_step."""
        logz, norm = model.conditional_logz(u_batch)
        nll = updates['free_energy'] + jnp.mean(-norm + logz)
        return {'nll': nll}

    @nnx.jit
    def _test_update_ext(self, model, test_u, test_v, updates):
        return self._train_step_ext(model, test_u, test_v, updates)


if __name__ == '__main__':
    with h5py.File('/data/iiyama/2dz2/recovery_learning.h5', 'r') as source:
        step0_data = source['plaq_data'][0][()]

    num_u = 10
    num_v = 10
    train_dataset = step0_data[:80_000]
    test_dataset = step0_data[80_000:]

    def train_model(num_h, batch_size, lr, idev, lock):
        with jax.default_device(jax.devices()[idev]):
            LOG.info('Training model with parameters num_h=%d batch_size=%d lr=%f on device %d',
                     num_h, batch_size, lr, idev)
            with lock:
                rngs = nnx.Rngs(params=0, sample=1)
                model = ConditionalRBM(num_u, num_v, num_h, rngs=rngs)
                callback = Callback()
                optax_fn = optax.adamw(learning_rate=lr)
                train_crbm(model, train_dataset=train_dataset[:batch_size],
                           test_dataset=test_dataset, batch_size=batch_size, num_epochs=1,
                           optax_fn=optax_fn, callback=callback)
            records = train_crbm(model, train_dataset=train_dataset, test_dataset=test_dataset,
                                 batch_size=batch_size, num_epochs=500, optax_fn=optax_fn,
                                 callback=callback)
            callback.as_arrays(records)
            weights = {key: np.array(getattr(model, key).value)
                       for key in ['weights_vu', 'weights_hu', 'weights_hv', 'bias_v', 'bias_h']}

        return weights, records

    with ThreadPoolExecutor() as executor:
        idev = 0
        futures = []
        lock = Lock()
        for batch_size in [40, 80, 160, 200, 320, 640][::-1]:
            for num_h in [64, 128, 256, 512][::-1]:
                for ilr, lr in enumerate([0.005, 0.001, 0.0005, 0.0001]):
                    futures.append((
                        executor.submit(train_model, num_h, batch_size, lr, idev, lock),
                        batch_size, num_h, ilr
                    ))
                    idev += 1
                    idev %= jax.device_count()

    with h5py.File('/data/iiyama/2dz2/hyperparam_search.h5', 'w') as out:
        for future, batch_size, num_h, ilr in futures:
            weights, records = future.result()
            group = out.create_group(f'b{batch_size}_h{num_h}_lr{ilr}')
            for key, value in weights.items():
                group.create_dataset(key, data=value)
            for key, value in records.items():
                group.create_dataset(key, data=value)
