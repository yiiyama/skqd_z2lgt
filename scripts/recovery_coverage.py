import os
import logging
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.preprocess import load_reco
from skqd_z2lgt.tasks.train_generator import load_model
from skqd_z2lgt.tasks.diagonalize import RandomExcitation

LOG = logging.getLogger(__name__)


@jax.jit
def to_indices_dupl(bits):
    shifts = 1 << jnp.arange(bits.shape[-1], dtype=np.int64)[::-1]
    return jnp.sum(bits * shifts, axis=-1)


def to_indices(bits):
    return np.array(jnp.unique(to_indices_dupl(bits)))


@nnx.jit
@nnx.scan(out_axes=nnx.Carry)
def coverage(carry, indices):
    probs, total = carry
    covered = probs.at[indices].get(mode='fill', wrap_negative_indices=False, fill_value=0.)
    return probs, total + jnp.sum(covered)


def coverage_ideal(parameters, trotter_probs, num_exp=100):
    @jax.jit(static_argnames=['nexp'])
    def get_coverage(key, probs, nexp):
        @nnx.scan(in_axes=nnx.Carry, length=nexp)
        def experiment(_key):
            _key, subkey = jax.random.split(_key)
            shots = parameters.runtime.shots_exp
            batch_size = parameters.crbm.gen_batch_size
            indices = jax.random.choice(_key, probs.shape[0], shape=(shots,), p=probs)
            indices = jnp.unique(indices, size=shots, fill_value=-1)
            return subkey, coverage((probs, 0.), indices.reshape((-1, batch_size)))[1]
        
        return experiment(key)[1]

    LOG.info('Computing ideal coverage..')
    key = jax.random.key(1234)
    coverages = np.empty((len(parameters.circuit.sampled_steps), num_exp))
    for istep, probs in enumerate(trotter_probs):
        coverages[istep] = get_coverage(key, probs, num_exp)

    return coverages


def probs_raw(parameters, trotter_probs):
    LOG.info('Obtaining probabilities covered by raw sampling..')
    probs = []
    for tprobs, step in zip(trotter_probs, parameters.circuit.sampled_steps):
        _, plaq_data = load_reco(parameters, 'exp', 0, step)
        indices = to_indices(plaq_data)
        probs.append(tprobs[indices])

    return probs


def coverage_local(parameters, trotter_probs, num_exp=100):
    @nnx.jit(static_argnames=['nexp'])
    def get_coverage(generator, plaq_data, probs, nexp):
        @nnx.scan
        def generate_fn(_generator, plaq_batch):
            size = (plaq_batch.shape[0], parameters.skqd.num_gen)
            flips = _generator.generate(size=size)
            shape = (np.prod(size), plaq_batch.shape[1])
            return _generator, (plaq_batch[:, None, :] ^ flips).reshape(shape)
        
        @nnx.scan(in_axes=nnx.Carry, length=nexp)
        def experiment(_generator):
            batch_size = parameters.crbm.gen_batch_size
            gen_states = generate_fn(_generator, plaq_data)[1].reshape((-1, plaq_data.shape[-1]))
            indices = to_indices_dupl(gen_states)
            indices = jnp.unique(indices, size=indices.shape[0], fill_value=-1)
            return _generator, coverage((probs, 0.), indices.reshape((-1, batch_size)))[1]
        
        return experiment(generator)[1]

    LOG.info('Computing coverage from quantum samples corrected with local bit flips..')
    rngs = nnx.Rngs(sample=1234)
    coverages = np.empty((len(parameters.circuit.sampled_steps), num_exp))
    for istep, (tprobs, step) in enumerate(zip(trotter_probs, parameters.circuit.sampled_steps)):
        _, plaq_data = load_reco(parameters, 'exp', 0, step)
        plaq_data = plaq_data.reshape((-1, parameters.crbm.gen_batch_size, plaq_data.shape[-1]))
        _, ref_data = load_reco(parameters, 'ref', 0, step)
        mean_activation = np.mean(ref_data, axis=0)
        generator = RandomExcitation(mean_activation, rngs=rngs)
        coverages[istep] = get_coverage(generator, plaq_data, tprobs, num_exp)

    return coverages


def coverage_crbm(parameters, trotter_probs, num_exp=100):
    @nnx.jit(static_argnames=['nexp'])
    def get_coverage(crbm_model, vtx_data, plaq_data, probs, nexp):
        @nnx.scan(in_axes=(nnx.Carry, 0, 0))
        def generate_fn(_crbm_model, vtx_batch, plaq_batch):
            num_gen = parameters.skqd.num_gen
            flips = _crbm_model.sample(vtx_batch, size=num_gen)
            shape = (plaq_batch.shape[0] * num_gen, plaq_batch.shape[1])
            return _crbm_model, (plaq_batch[None, :, :] ^ flips).reshape(shape)
        
        @nnx.scan(in_axes=nnx.Carry, length=nexp)
        def experiment(_crbm_model):
            batch_size = parameters.crbm.gen_batch_size
            gen_states = generate_fn(_crbm_model, vtx_data, plaq_data)[1]
            gen_states = gen_states.reshape((-1, plaq_data.shape[-1]))
            indices = to_indices_dupl(gen_states)
            indices = jnp.unique(indices, size=indices.shape[0], fill_value=-1)
            return _crbm_model, coverage((probs, 0.), indices.reshape((-1, batch_size)))[1]
        
        return experiment(crbm_model)[1]

    LOG.info('Computing coverage from quantum samples corrected with CRBM..')
    coverages = np.empty((len(parameters.circuit.sampled_steps), num_exp))
    for istep, (tprobs, step) in enumerate(zip(trotter_probs, parameters.circuit.sampled_steps)):
        vtx_data, plaq_data = load_reco(parameters, 'exp', 0, step)
        batched_shape = (-1, parameters.crbm.gen_batch_size)
        plaq_data = plaq_data.reshape(batched_shape + (plaq_data.shape[-1],))
        vtx_data = vtx_data.reshape(batched_shape + (vtx_data.shape[-1],))
        model = load_model(parameters, 0, step)
        coverages[istep] = get_coverage(model, vtx_data, plaq_data, tprobs, num_exp)

    return coverages


def main(pkgpath):
    parameters = open_output(pkgpath)

    LOG.info('Loading trotter_probs..')
    trotter_probs = []
    for step in parameters.circuit.sampled_steps:
        path = Path(parameters.pkgpath) / f'trotter_idt0_s{step}.h5'
        with h5py.File(str(path), 'r') as source:
            trotter_probs.append(source['probs'][()])

    with h5py.File(Path(pkgpath) / 'recovery_coverage.h5', 'a') as out_file:
        if 'coverage_ideal' not in out_file:
            coverages = coverage_ideal(parameters, trotter_probs)
            out_file.create_dataset('coverage_ideal', data=coverages)
        if 'probs_raw' not in out_file:
            probs = probs_raw(parameters, trotter_probs)
            group = out_file.create_group('probs_raw')
            for pr, step in zip(probs, parameters.circuit.sampled_steps):
                group.create_dataset(f's{step}', data=pr)
        if 'coverage_local' not in out_file:
            coverages = coverage_local(parameters, trotter_probs)
            out_file.create_dataset('coverage_local', data=coverages)
        if 'coverage_crbm' not in out_file:
            coverages = coverage_crbm(parameters, trotter_probs)
            out_file.create_dataset('coverage_crbm', data=coverages)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--gpu')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    main(options.pkgpath)
