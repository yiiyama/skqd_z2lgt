"""Train the CRBM for configuration recovery."""
import os
from functools import partial
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from skqd_z2lgt.recovery.crbm import ConditionalRBM
from skqd_z2lgt.recovery.loss import cd_meanloss
from skqd_z2lgt.recovery.train import train_crbm
from skqd_z2lgt.recovery.metric import success_rate
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.io import load_reco


def save_model(
    parameters: Parameters,
    model: ConditionalRBM,
    records: dict[str, np.ndarray],
    idt: int,
    step: int,
    logger: Optional[logging.Logger] = None
):
    """Save the weight parameters of the CRBM model into file."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving CRBM model for dt=%f step=%d', parameters.circuit.dts[idt], step)

    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{step}.h5'
    os.makedirs(path.parent, exist_ok=True)
    with h5py.File(path, 'w', libver='latest') as out:
        model.save(out)
        group = out.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)


def load_model(
    parameters: Parameters,
    idt: int,
    step: int,
    compile_for: Optional[tuple[int, int]] = None,
    jax_device_id: Optional[int] = None
) -> ConditionalRBM:
    """Construct CRBM models from saved weights."""
    if jax_device_id is None:
        device = None
    else:
        device = jax.devices()[jax_device_id]

    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{step}.h5'
    with h5py.File(path, 'r', libver='latest') as source:
        with jax.default_device(device):
            model = ConditionalRBM.load(source)
            if compile_for:
                model.sample(
                    jnp.zeros((compile_for[0], model.weights_vu.shape[1]), dtype=np.uint8),
                    size=compile_for[1]
                )
    return model


def train_single_model(
    parameters: Parameters,
    idt: int,
    step: int,
    seed_params: int = 1234,
    seed_sample: int = 5678,
    logger: Optional[logging.Logger] = None
):
    """Train a single CRBM model."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Training CRBM model for time interval %f, Trotter step %d',
                parameters.circuit.dts[idt], step)
    cparams = parameters.crbm

    data = load_reco(parameters, etype='ref', idt=idt, step=step)
    rngs = nnx.Rngs(params=seed_params, sample=seed_sample)
    model = ConditionalRBM(data[0].shape[-1], data[1].shape[-1], cparams.num_h, rngs=rngs)
    loss_fn_train = partial(cd_meanloss, impl='sum', unroll=20)
    loss_fn_test = partial(cd_meanloss, impl='tensordot', unroll=50)
    metrics_test = {'success_rate': (nnx.metrics.Average, success_rate(parameters.skqd.num_gen))}

    def best_model_cond(iepoch, records):
        sr = records['test_success_rate']
        n_epochs = sr.shape[0]
        return jnp.all(jnp.logical_or(sr < sr[iepoch], jnp.arange(n_epochs) >= iepoch))

    def convergence(iepoch, records):
        sr = records['test_success_rate']
        i_best = jnp.argmax(sr)
        return iepoch > i_best + 20

    optax_fn = optax.adamw(learning_rate=cparams.learning_rate, weight_decay=cparams.weight_decay)

    best_model, records = train_crbm(model, data, batch_size=cparams.train_batch_size,
                                     num_epochs=cparams.num_epochs, loss_fn_train=loss_fn_train,
                                     loss_fn_test=loss_fn_test, optax_fn=optax_fn,
                                     metrics_test=metrics_test, best_model_cond=best_model_cond,
                                     convergence=convergence,
                                     train_eval_unit=cparams.train_eval_unit)

    save_model(parameters, best_model, records, idt, step, logger=logger)

    return best_model, records
