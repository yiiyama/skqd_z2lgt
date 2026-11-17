# pylint: disable=no-member
"""SKQD with configuration recovery."""
import logging
import math
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from rqutils.paulis.symplectic import PauliSumXZ
from skqd_z2lgt.truncation_sqd import (
    ProductStateSampler,
    make_pt_core,
    generate_recovery,
    generate_random,
    iterative_sqd
)
from skqd_z2lgt.recovery.crbm import ConditionalRBM
from skqd_z2lgt.recovery.extensions import vertical_reflection
from skqd_z2lgt.orchestration.common import make_lattice
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.common import make_dual_lattice
from skqd_z2lgt.orchestration.io import (
    load_reco,
    save_skqd_result,
    initialize_skqd_snapshot,
    save_skqd_snapshot,
    load_skqd_snapshot
)
from skqd_z2lgt.orchestration.train_generator import load_model, new_model
try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


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


def prepare_shape_struct(parameters: Parameters) -> tuple[jax.Array]:
    """Prepare a dummy array to extract the shape from."""
    shots = parameters.runtime.shots_exp
    batch_size = parameters.crbm.gen_batch_size
    num_batches = int(np.ceil(shots / batch_size).astype(int))
    num_datasets = len(parameters.circuit.dts) * len(parameters.circuit.sampled_steps)
    return (jax.ShapeDtypeStruct((num_datasets, num_batches, batch_size), np.uint8),)


def prepare_crbm_models(parameters: Parameters) -> ConditionalRBM:
    """Return stacked (as pytrees) CRBM models."""
    crbm_models = []
    for idt in range(len(parameters.circuit.dts)):
        for step in parameters.circuit.sampled_steps:
            crbm_models.append(load_model(parameters, idt, step))

    crbm_models = jax.tree.map(lambda *xs: jnp.stack(xs), *crbm_models)
    return crbm_models


def prepare_local_generators(parameters: Parameters) -> ConditionalRBM:
    """Return stacked (as pytrees) RandomExcitation generators."""
    ref_data = load_reco(parameters, etype='ref')
    crbm_models = []
    for idt in range(len(parameters.circuit.dts)):
        for istep, step in enumerate(parameters.circuit.sampled_steps):
            model = new_model(parameters, idt, step, num_h=1)
            for pname in ['weights_vu', 'weights_hu', 'weights_hv', 'bias_h']:
                params = getattr(model, pname)
                params.set_value(jnp.zeros_like(params.get_value()))
            mean_activation = np.mean(ref_data[idt][istep][1], axis=0)
            model.bias_v.set_value(-jnp.log(1. / mean_activation - 1.))
            model.therm_steps = 0
            crbm_models.append(model)

    crbm_models = jax.tree.map(lambda *xs: jnp.stack(xs), *crbm_models)
    return crbm_models


def prepare_random_generators(parameters: Parameters) -> ProductStateSampler:
    """Return stacked (as pytrees) RandomExcitation generators."""
    lattice = make_lattice(parameters)
    num_qubits = lattice.num_active_plaquettes
    generators = []
    for idt, dt in enumerate(parameters.circuit.dts):
        for istep in range(len(parameters.circuit.sampled_steps)):
            seed_sample = parameters.crbm.seed_sample
            seed_sample += idt * len(parameters.circuit.sampled_steps) + istep
            rngs = nnx.Rngs(sample=seed_sample)
            angle = -2. * parameters.lgt.plaquette_energy * dt
            sampler = ProductStateSampler(num_qubits, angle, rngs=rngs)
            generators.append(sampler)

    generators = jax.tree.map(lambda *xs: jnp.stack(xs), *generators)
    return generators


def load_snapshot(parameters: Parameters, mode: str, generators: ConditionalRBM):
    if MPI and jax.process_index() != 0:
        MPI.COMM_WORLD.bcast(None, root=0)
    iteration, history, carryover, counts = load_skqd_snapshot(parameters, mode)
    # If max_iterations has been updated, resize the arrays and re-initialize the snapshot file
    if jax.process_index() == 0:
        if history[0].shape[0] != parameters.skqd.max_iterations + 1:
            history = tuple(np.resize(rec, parameters.skqd.max_iterations + 1)
                            for rec in history)
            initialize_skqd_snapshot(parameters, mode)
            save_skqd_snapshot(parameters, mode, iteration, history, carryover, counts)
        if MPI and MPI.COMM_WORLD.Get_size() != 1:
            MPI.COMM_WORLD.bcast('ready', root=0)
        
    generators.rngs.sample.count.set_value(counts)
    return (iteration + 1, history, carryover)


def diagonalize_in_mode(
    parameters: Parameters,
    mode: str,  # 'cr', 'lo', or 'cl'
    resume: bool = False,
    profile: Optional[str] = None,    
    logger: Optional[logging.Logger] = None
) -> float:
    """Project and diagonalize the Hamiltonian with CRBM or local configuration recovery."""
    logger = logger or logging.getLogger(__name__)
    proc_id = jax.process_index()

    def loginfo(msg, *args):
        if proc_id == 0:
            logger.info(msg, *args)

    if mode == 'cr':
        loginfo('Loading and stacking CRBM models')
        generators = prepare_crbm_models(parameters)
        generator_fn = generate_recovery
        generator_args = prepare_data(parameters)
    elif mode == 'lo':
        loginfo('Using the mean of reference circuit data as single-plaquette probability')
        generators = prepare_local_generators(parameters)
        generator_fn = generate_recovery
        generator_args = prepare_data(parameters)
    elif mode == 'cl':
        loginfo('Sampling from product states')
        generators = prepare_random_generators(parameters)
        generator_fn = generate_random
        generator_args = prepare_shape_struct(parameters)

    loginfo('Setting up the Hamiltonian')
    dual_lattice = make_dual_lattice(parameters)
    hamiltonian = PauliSumXZ.from_paulisum(
        dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy),
        add_padding=True
    )
    loginfo('Generating the PT core (<=%d excitations)', parameters.skqd.pt_core_max)
    pt_core = make_pt_core(hamiltonian.num_qubits, parameters.skqd.pt_core_max)

    extension_fns = []
    for fname in parameters.skqd.extensions:
        loginfo('Setting up CR extension %s', fname)
        # Potentially set up a dict with factory functions
        if fname == 'vertical_reflection':
            extension_fns.append(vertical_reflection(dual_lattice))
    
    extension = None
    if extension_fns:
        def extension(states_p):
            for fn in extension_fns:
                states_p = fn(states_p)
            return states_p

    def callback(iteration, result, history, gens):
        if proc_id == 0:
            save_skqd_snapshot(parameters, mode, iteration, history, result[2],
                               gens.rngs.sample.count.get_value())

    if resume:
        init = load_snapshot(parameters, mode, generators)
    else:
        initialize_skqd_snapshot(parameters, mode)
        init = None

    loginfo('Starting SKQD iterations. Max iterations %d, convergence tolerance %.1e',
            parameters.skqd.max_iterations, parameters.skqd.energy_convergence)
    
    if profile:
        jax.profiler.start_trace(profile)
    niter, history = iterative_sqd(hamiltonian, generator_fn, generators, generator_args,
                                   parameters.skqd.num_gen, pt_core, parameters.skqd.carryover,
                                   parameters.skqd.max_iterations,
                                   parameters.skqd.energy_convergence,
                                   extension=extension, init=init, callback=callback)
    if profile:
        jax.profiler.stop_trace()

    energies = history[0][:niter]
    subspace_dims = history[1][:niter]
    if proc_id == 0:
        save_skqd_result(parameters, mode, energies, subspace_dims)

    loginfo('Diagonalization completed after %d iterations. Final energy %f, subspace dim %d',
            niter, energies[-1], subspace_dims[-1])

    return float(energies[-1])


def get_expected_footprint(parameters: Parameters):
    """Compute the expected device RAM usage in bytes."""
    gen_size = (len(parameters.circuit.dts) * len(parameters.circuit.sampled_steps)
                * parameters.runtime.shots_exp * parameters.skqd.num_gen)
    for fname in parameters.skqd.extensions:
        if fname == 'vertical_reflection':
            gen_size *= 2
    num_p = make_lattice(parameters).num_active_plaquettes
    pt_core_size = sum(math.comb(num_p, i) for i in range(parameters.skqd.pt_core_max + 1))
    # Size of the SQD subspace N
    subspace_size = gen_size + parameters.skqd.carryover + pt_core_size
    # 4 bytes per index, N indices per X signature, num_p + 1 X signatures
    xcache_size = 4 * subspace_size * (num_p + 1)
    # 8 bytes per entry, N entries per materialization, 9+ materializations (depends on sharding)
    vectors_size = 8 * subspace_size * 18
    # ⌈(nq+1)/8⌉ bytes per state, N+pt_core states
    states_size = np.ceil((num_p + 1) / 8).astype(int) * (subspace_size + pt_core_size)
    return xcache_size + vectors_size + states_size


if __name__ == '__main__':
    import os
    import sys
    import time
    import argparse
    from skqd_z2lgt.distributed import direct_sharding
    from skqd_z2lgt.orchestration.open_output import open_output

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('mode')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--niter', type=int)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--profile')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, options.log_level.upper()),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    if (jobid := os.getenv('PBS_JOBID')):
        print(f'PBS JOBID {jobid} started at {time.asctime()}')

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    direct_sharding(options.mpi, options.gpus)
    parameters = open_output(options.pkgpath)
    if options.niter:
        parameters.skqd.max_iterations = options.niter
    diagonalize_in_mode(parameters, options.mode, resume=options.resume, profile=options.profile)
