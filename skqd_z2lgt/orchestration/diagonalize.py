# pylint: disable=no-member
"""SKQD with configuration recovery."""
import logging
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from flax import nnx
from rqutils.paulis.symplectic import PauliSumXZ
from skqd_z2lgt.perturbative_sqd import make_pt_core, iterative_sqd
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.common import make_dual_lattice
from skqd_z2lgt.orchestration.io import load_reco, save_skqd_result
from skqd_z2lgt.orchestration.train_generator import load_model


def prepare_data(parameters: Parameters) -> tuple[jax.Array, jax.Array]:
    """Pad and reshape the data to be batch-processable and return as stacks."""
    shots = parameters.runtime.shots_exp
    batch_size = parameters.crbm.gen_batch_size
    num_batches = int(np.ceil(shots / batch_size).astype(int))
    residue = num_batches * batch_size - shots

    exp_data = load_reco(parameters, etype='exp')
    syndrome_data = []
    plaq_data = []
    for idt in range(len(parameters.circuit.dts)):
        for istep in range(len(parameters.circuit.sampled_steps)):
            syndrome, plaq = exp_data[idt][istep]
            if residue != 0:
                syndrome = np.pad(syndrome, [(0, residue), (0, 0)])
                plaq = np.pad(plaq, [(0, residue), (0, 0)])
            syndrome_data.append(syndrome.reshape((num_batches, batch_size, -1)))
            plaq_data.append(plaq.reshape((num_batches, batch_size, -1)))

    syndrome_data = jnp.stack(syndrome_data)
    plaq_data = jnp.stack(plaq_data)
    return syndrome_data, plaq_data


def prepare_crbm_models(parameters: Parameters) -> nnx.Module:
    """Return stacked (as pytrees) CRBM models."""
    crbm_models = []
    for idt in range(len(parameters.circuit.dts)):
        for step in parameters.circuit.sampled_steps:
            crbm_models.append(
                load_model(parameters, idt, step)
            )

    crbm_models = jax.tree.map(lambda *xs: jnp.stack(xs), *crbm_models)
    return crbm_models


class RandomExcitation(nnx.Module):
    """Random plaquette excitation generator."""
    def __init__(self, mean_activation: np.ndarray, *, rngs: nnx.Rngs):
        self.probs = nnx.Param(mean_activation)
        self.rngs = rngs

    def sample(self, batch: NDArray, size: int) -> jax.Array:
        shape = (size, batch.shape[0]) + self.probs.shape
        uvalues = jax.random.uniform(self.rngs.sample(), shape)
        return jnp.asarray(uvalues < self.probs[...], dtype=np.uint8)


def prepare_rand_generators(parameters: Parameters) -> RandomExcitation:
    """Return stacked (as pytrees) RandomExcitation generators."""
    ref_data = load_reco(parameters, etype='ref')
    generators = []
    seed = 1234
    for idt in range(len(parameters.circuit.dts)):
        for istep in range(len(parameters.circuit.sampled_steps)):
            rngs = nnx.Rngs(sample=seed)
            seed += 1
            mean_activation = np.mean(ref_data[idt][istep][1], axis=0)
            generators.append(RandomExcitation(mean_activation, rngs=rngs))

    generators = jax.tree.map(lambda *xs: jnp.stack(xs), *generators)
    return generators


def diagonalize_in_mode(
    parameters: Parameters,
    mode: str,  # 'cr' or 'rn'
    logger: Optional[logging.Logger] = None
) -> float:
    """Project and diagonalize the Hamiltonian with configuration recovery or random bitflips."""
    logger = logger or logging.getLogger(__name__)

    logger.info('Padding and stacking data')
    syndrome_data, plaq_data = prepare_data(parameters)
    if mode == 'cr':
        logger.info('Loading and stacking CRBM models')
        generators = prepare_crbm_models(parameters)
    else:
        logger.info('Using the mean of reference circuit data as single-plaquette probability')
        generators = prepare_rand_generators(parameters)

    logger.info('Setting up the Hamiltonian')
    dual_lattice = make_dual_lattice(parameters)
    hamiltonian = PauliSumXZ.from_paulisum(
        dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy),
        add_padding=True
    )
    logger.info('Generating the PT core (<=%d excitations)', parameters.skqd.pt_core_max)
    pt_core = make_pt_core(plaq_data.shape[-1], parameters.skqd.pt_core_max)
    logger.info('Starting SKQD iterations')
    energies, subspace_dims = iterative_sqd(hamiltonian, generators, syndrome_data, plaq_data,
                                            parameters.skqd.num_gen, pt_core,
                                            parameters.skqd.max_iterations,
                                            parameters.skqd.energy_convergence)

    if jax.process_index() == 0:
        save_skqd_result(parameters, mode, energies, subspace_dims)

    return float(energies[-1])


if __name__ == '__main__':
    import os
    import sys
    import argparse
    from jax.sharding import AxisType
    from skqd_z2lgt.orchestration.open_output import open_output

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('mode')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stdout)
    logging.getLogger('jax').setLevel(logging.WARNING)

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    if options.mpi:
        jax.distributed.initialize(cluster_detection_method='mpi4py')
    if (ndev := jax.device_count()) > 1:
        jax.set_mesh(jax.make_mesh((ndev,), ('x',), (AxisType.Explicit,)))

    diagonalize_in_mode(open_output(options.pkgpath), options.mode)
