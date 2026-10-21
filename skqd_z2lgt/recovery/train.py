# pylint: disable=unused-argument
"""Routine for training the CRBM."""
from collections.abc import Callable
import logging
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from skqd_z2lgt.recovery.crbm import ConditionalRBM
from skqd_z2lgt.recovery.loss import LossFn, cd_meanloss

LOG = logging.getLogger(__name__)


def make_train_fn(grad_fn, metric_fns):
    def train_step(carry, data):
        """Compute the gradient on a single minibatch and update the parameters.

        Also computes loss and other metrics to monitor the training progress.
        """
        model, optimizer, metrics = carry
        u_batch, v_batch = data
        loss, grads = grad_fn(model, u_batch, v_batch)
        updates = {'loss': loss.astype(np.float32)}
        for name, fn in metric_fns.items():
            updates[name] = fn(model, u_batch, v_batch, updates).astype(np.float32)
        metrics.update(**updates)
        optimizer.update(model, grads)
        return carry

    def train_unit(carry, data):
        """Scan over `eval_unit` minibatches and compute the averages of the metrics."""
        nnx.scan(train_step, out_axes=nnx.Carry)(carry, data)
        metrics = carry[2]
        records = metrics.compute()
        metrics.reset()
        return carry, records

    def train_epoch(model, optimizer, metrics, data):
        """Scan over the entire train epoch in steps of `eval_unit`."""
        return nnx.scan(train_unit)((model, optimizer, metrics), data)[1]

    return train_epoch


def make_test_fn(loss_fn, metric_fns):
    def compute_batch(carry, data):
        u_batch, v_batch = data
        model, metrics = carry
        updates = {'loss': loss_fn(model, u_batch, v_batch).astype(np.float32)}
        for name, fn in metric_fns.items():
            updates[name] = fn(model, u_batch, v_batch, updates).astype(np.float32)
        metrics.update(**updates)
        return carry

    def compute_test_metrics(model, metrics, data):
        nnx.scan(compute_batch, out_axes=nnx.Carry)((model, metrics), data)
        records = metrics.compute()
        metrics.reset()
        return records

    return compute_test_metrics


def make_train_loop(train_epoch, compute_test_metrics, best_cond, convergence, n_epochs, tr_shape):
    def loop_body(val):
        iepoch, best_model = val[:2]
        records, model, optimizer, metrics_train, metrics_test, data, data_test, rngs = val[2:]
        if LOG.getEffectiveLevel() <= logging.INFO:
            jax.debug.print('Starting epoch {}/{}', iepoch, n_epochs)

        indices_train = rngs.permutation(jnp.arange(np.prod(tr_shape)))
        indices_train = indices_train.reshape(tr_shape)
        data_train = tuple(d[indices_train] for d in data)

        records_epoch = train_epoch(model, optimizer, metrics_train, data_train)
        for name, arr in records_epoch.items():
            records[f'train_{name}'] = records[f'train_{name}'].at[iepoch].set(arr)

        test_values = compute_test_metrics(model, metrics_test, data_test)
        for name, value in test_values.items():
            records[f'test_{name}'] = records[f'test_{name}'].at[iepoch].set(value)

        def make_copy():
            if LOG.getEffectiveLevel() <= logging.INFO:
                jax.debug.print('Found best model so far with test values {}', test_values)
            return model.copy()

        best_model = jax.lax.cond(
            best_cond(iepoch, records),
            make_copy,
            lambda: best_model
        )

        return (iepoch + 1, best_model) + val[2:]

    @nnx.jit
    def train_loop(model, optimizer, metrics_train, metrics_test, data, data_test, rngs):
        records = {f'train_{name}': jnp.empty((n_epochs, tr_shape[0]))
                    for name in ['loss'] + metrics_train._metric_names}
        records |= {f'test_{name}': jnp.empty(n_epochs)
                    for name in ['loss'] + metrics_test._metric_names}
        return jax.lax.while_loop(
            lambda val: (val[0] < n_epochs) & ~convergence(val[0], val[2]),
            loop_body,
            (0, model, records, model, optimizer, metrics_train, metrics_test, data, data_test, rngs)
        )[:3]

    return train_loop


def train_crbm(
    model: ConditionalRBM,
    data: tuple[np.ndarray, np.ndarray],
    train_fraction: float = 0.9,
    batch_size: int = 64,
    num_epochs: int = 100,
    loss_fn_train: Optional[LossFn] = None,
    loss_fn_test: Optional[LossFn] = None,
    optax_fn: Optional[Callable] = None,
    metrics_train: Optional[dict[str, tuple]] = None,
    metrics_test: Optional[dict[str, tuple]] = None,
    best_model_cond: Optional[Callable[[int, dict[str, np.ndarray]], bool]] = None,
    convergence: Optional[Callable[[int, dict[str, np.ndarray]], bool]] = None,
    train_eval_unit: int = 100,
    test_batch_size: int = 1000,
    seed: int = 0
):
    loss_fn_train = loss_fn_train or cd_meanloss
    loss_fn_test = loss_fn_test or loss_fn_train
    grad_fn = nnx.value_and_grad(loss_fn_train)
    best_model_cond = best_model_cond or (lambda i, r: True)
    convergence = convergence or (lambda i, r: False)

    metrics_train = metrics_train or {}
    metric_fns_train = {name: fn for name, (_, fn) in metrics_train.items()}
    metrics_test = metrics_test or metrics_train
    metric_fns_test = {name: fn for name, (_, fn) in metrics_test.items()}

    size_full = data[0].shape[0]

    size_train = int(np.ceil(size_full * train_fraction).astype(int))
    ebatch_size = train_eval_unit * batch_size
    num_ebatches = size_train // ebatch_size
    if num_ebatches == 0:
        num_ebatches = 1
        train_eval_unit = size_train // batch_size
        ebatch_size = train_eval_unit * batch_size
        LOG.warning('Number of training samples %d is smaller than the requested ebatch size. Using'
                    ' num_ebatches=1, train_eval_unit=%d, ebatch_size=%d instead.',
                    size_train, train_eval_unit, ebatch_size)
    size_train = num_ebatches * ebatch_size
    train_shape = (num_ebatches, train_eval_unit, batch_size)

    size_test = size_full - size_train
    num_test_batches = size_test // test_batch_size
    if num_test_batches == 0:
        LOG.warning('Number of test samples %d is smaller than the requested batch size. Using'
                    ' num_test_batches=1, test_batch_size=%d instead.',
                    size_test, test_batch_size)
        num_test_batches = 1
        test_batch_size = size_test
    size_test = num_test_batches * test_batch_size
    indices_test = np.arange(size_full - size_test, size_full)
    indices_test = indices_test.reshape(num_test_batches, test_batch_size)
    data_test = tuple(jax.device_put(d[indices_test]) for d in data)

    LOG.info('Training data shape %s=%d, test data size %d', train_shape, size_train, size_test)

    optimizer = nnx.Optimizer(model, optax_fn or optax.adamw(learning_rate=0.001, weight_decay=0.1),
                              wrt=nnx.Param)

    metric_mods_train = nnx.metrics.MultiMetric(
        **({'loss': nnx.metrics.Average('loss')}
           | {name: cls(name) for name, (cls, _) in metrics_train.items()})
    )
    metric_mods_test = nnx.metrics.MultiMetric(
        **({'loss': nnx.metrics.Average('loss')}
           | {name: cls(name) for name, (cls, _) in metrics_test.items()})
    )

    train_loop = make_train_loop(
        make_train_fn(grad_fn, metric_fns_train),
        make_test_fn(loss_fn_test, metric_fns_test),
        best_model_cond,
        convergence,
        num_epochs,
        train_shape
    )

    nepochs, best_model, records = train_loop(model, optimizer, metric_mods_train, metric_mods_test,
                                              data, data_test, nnx.Rngs(seed))
    for key, value in records.items():
        records[key] = np.array(value[:nepochs])

    return best_model, records
