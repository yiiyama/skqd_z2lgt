import os
import asyncio
from pathlib import Path
import logging
import json
import tempfile
import math
import hashlib
from prefect import flow, get_run_logger
from prefect.assets import materialize
from prefect.cache_policies import CacheKeyFnPolicy
from prefect.context import get_run_context
from prefect.filesystems import LocalFileSystem
from prefect.variables import Variable
from qcsc_prefect_executor.from_blocks import run_job_from_blocks
from qcsc_prefect_blocks.common import HPCProfileBlock
from qcsc_prefect.integrations.qiskit import QiskitRuntimeConfig
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.open_output import open_output, open_from_yaml
from skqd_z2lgt.orchestration.sample_quantum import (
    get_trotter_circuits,
    filter_out_completed,
    run_sampler
)
from skqd_z2lgt.orchestration.diagonalize import get_expected_footprint
from skqd_z2lgt.orchestration.io import load_dmrg, load_skqd_result


@flow(persist_result=True, result_serializer='json')
async def skqd_z2lgt(parameters: Parameters) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD."""
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    open_output(parameters, logger)
    suffices = [f'dt{idt}_k{step}'
                for idt in range(len(parameters.circuit.dts))
                for step in parameters.circuit.sampled_steps]
    raw_filenames = [f'data/raw/{e}_{s}.h5' for e in ['exp', 'ref'] for s in suffices]
    reco_exp_filenames = [f'data/reco/exp_{s}.h5' for s in suffices]
    reco_ref_filenames = [f'data/reco/ref_{s}.h5' for s in suffices]
    reco_filenames = reco_exp_filenames + reco_ref_filenames
    crbm_filenames = [f'crbm/{s}.h5' for s in suffices]

    result_storage = await get_result_storage(parameters)
    cache_policy = CacheKeyFnPolicy(cache_key_fn=assert_assets).configure(
        key_storage=Path(result_storage.basepath) / 'cache_keys'
    )

    if parameters.dmrg:
        dmrg_future = dmrg.with_options(
            assets=[(Path(parameters.pkgpath) / 'dmrg.h5').as_uri()],
            cache_policy=cache_policy,
            result_storage=result_storage
        ).submit(parameters)
    
    await sample_quantum.with_options(
        assets=[(Path(parameters.pkgpath) / f).as_uri() for f in raw_filenames],
        cache_policy=cache_policy,
        result_storage=result_storage
    )(parameters)

    await preprocess.with_options(
        assets=[(Path(parameters.pkgpath) / f).as_uri() for f in reco_filenames],
        asset_deps=[(Path(parameters.pkgpath) / f).as_uri() for f in raw_filenames],
        cache_policy=cache_policy,
        result_storage=result_storage
    )(parameters)

    await train_generators.with_options(
        assets=[(Path(parameters.pkgpath) / f).as_uri() for f in crbm_filenames],
        asset_deps=[(Path(parameters.pkgpath) / f).as_uri() for f in reco_ref_filenames],
        cache_policy=cache_policy,
        result_storage=result_storage
    )(parameters)

    energy_cr = await diagonalize.with_options(
        assets=[(Path(parameters.pkgpath) / 'skqd_cr.h5').as_uri()],
        asset_deps=[(Path(parameters.pkgpath) / f).as_uri() for f in reco_exp_filenames],
        cache_policy=cache_policy,
        result_storage=result_storage
    )(parameters)

    if parameters.dmrg:
        dmrg_energy = dmrg_future.result()
        logger.info('DMRG energy: %f', dmrg_energy)
    # logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy (full conf. recovery): %f', energy_cr)

    return energy_cr


async def get_result_storage(parameters: Parameters):
    block_name = hashlib.md5(parameters.pkgpath.encode('utf-8')).hexdigest()
    try:
        storage = await LocalFileSystem.load(block_name)
    except ValueError:
        storage = LocalFileSystem(basepath=str(Path(parameters.pkgpath) / 'results'))
        await storage.save(block_name)

    return storage


def assert_assets(context, inputs):
    """Remove the cache key file if any asset is missing."""
    cache_key = context.task.name
    # Path.from_uri() is available only for python>=3.13
    if not all(os.path.exists(a.key.replace('file://', '')) for a in context.task.assets):
        try:
            os.unlink(Path(context.task.cache_policy.key_storage) / cache_key)
        except FileNotFoundError:
            pass

    return cache_key


@materialize('file://')
async def dmrg(parameters: Parameters):
    result = await run_job_from_blocks(
        command_block_name=parameters.python_cmd,
        execution_profile_block_name=parameters.python_cpu_single_exec_profile,
        hpc_profile_block_name=parameters.hpc_profile,
        work_dir=Path(parameters.pkgpath) / 'work/dmrg',
        user_args=[str(Path(__file__).parents[0] / 'dmrg.py'), parameters.pkgpath],
        script_filename='run',
        execution_profile_overrides={'walltime': '01:00:00'}
    )
    if result.exit_status != 0:
        raise RuntimeError(f"DMRG job failed: exit_status={result.exit_status}")

    return load_dmrg(parameters)


@materialize('file://')
async def sample_quantum(parameters: Parameters):
    logger = get_run_logger()
    context = get_run_context()

    runtime_config = await QiskitRuntimeConfig.load(parameters.runtime.config_block)
    # QiskitRuntimeConfig.get_backend() currently does not support use_fractional_gates option, so
    # we go through the service
    service = runtime_config.get_service()
    backend = service.backend(runtime_config.backend_name, use_fractional_gates=True)
    layout, circuits = get_trotter_circuits(parameters, backend.target, logger=logger)
    if not parameters.circuit.layout:
        parameters.circuit.layout = layout
        with open(Path(parameters.pkgpath) / 'parameters.json', 'w', encoding='utf-8') as out:
            out.write(parameters.model_dump_json())

    try:
        with open(Path(context.result_store.result_storage.basepath) / context.task.name) as src:
            task_map = json.read(src)
    except FileNotFoundError:
        task_map = {}

    circuits = filter_out_completed(parameters, circuits, task_map, service, logger=logger)
    task_map |= run_sampler(parameters, circuits, backend, logger=logger)
    
    return task_map


@materialize('file://', asset_deps=[])
async def preprocess(parameters: Parameters):
    result = await run_job_from_blocks(
        command_block_name=parameters.python_cmd,
        execution_profile_block_name=parameters.python_cpu_single_exec_profile,
        hpc_profile_block_name=parameters.hpc_profile,
        work_dir=Path(parameters.pkgpath) / 'work/preprocess',
        user_args=[str(Path(__file__).parents[0] / 'preprocess.py'), parameters.pkgpath],
        script_filename='run',
        execution_profile_overrides={'walltime': '00:30:00'}
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Preprocess job failed: exit_status={result.exit_status}")

    return result.exit_status


@materialize('file://', asset_deps=[])
async def train_generators(parameters: Parameters):
    logger = get_run_logger()
    task = get_run_context().task

    gpus = (await Variable.get(parameters.hpc_profile)).get('gpus')
    if isinstance(gpus, str):
        gpus = list(map(int, gpus.split(',')))

    futures = {}
    igpu = 0
    for idt in range(len(parameters.circuit.dts)):
        for step in parameters.circuit.sampled_steps:
            if os.path.exists(Path(parameters.pkgpath) / f'crbm/dt{idt}_k{step}.h5'):
                logger.info('CRBM weights for dt%d_k%d exist', idt, step)
                continue

            add_args = []
            if gpus:
                add_args += ['--gpu', str(gpus[igpu])]
                igpu = (igpu + 1) % len(gpus)

            future = train_single_generator.with_options(
                assets=[(Path(parameters.pkgpath) / f'crbm/dt{idt}_k{step}.h5').as_uri()],
                asset_deps=[(Path(parameters.pkgpath) / f'data/reco/dt{idt}_k{step}.h5').as_uri()],
                cache_policy=task.cache_policy,
                name=f'train_single_generator_dt{idt}_k{step}',
                result_storage=task.result_storage
            ).submit(parameters, idt, step, add_args)
            futures[(idt, step)] = future

    exit_statuses = {key: fut.result() for key, fut in futures.items()}
    if any(s != 0 for s in exit_statuses.values()):
        raise RuntimeError(f"Train generators job failed: exit_statuses={exit_statuses}")

    return 0


@materialize('file://', asset_deps=[])
async def train_single_generator(parameters: Parameters, idt: int, step: int, add_args: list[str]):
    result = await run_job_from_blocks(
        command_block_name=parameters.python_cuda_cmd,
        execution_profile_block_name=parameters.python_gpu_single_exec_profile,
        hpc_profile_block_name=parameters.hpc_profile,
        work_dir=Path(parameters.pkgpath) / 'work' / get_run_context().task.name,
        user_args=[
            str(Path(__file__).parents[0] / 'train_generator.py'),
            parameters.pkgpath,
            '--idt', f'{idt}',
            '--step', f'{step}'
        ] + add_args,
        script_filename='run',
        execution_profile_overrides={'walltime': '02:00:00'}
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Train generator ({idt}, {step}) job failed:"
                           f" exit_status={result.exit_status}")

    return result.exit_status


@materialize('file://', asset_deps=[])
async def diagonalize(parameters: Parameters):
    logger = get_run_logger()
    
    # Determine the required number of nodes
    
    gpu_memory = (await Variable.get(parameters.hpc_profile)).get('gpu_memory_gib', 80) *  (2 ** 30)
    footprint = get_expected_footprint(parameters)
    # Give a comfort factor of 1.2
    minimum_num_nodes = footprint * 1.2 / gpu_memory
    num_devices = 2 ** int(math.ceil(math.log2(minimum_num_nodes)))
    logger.info('Expected footprint %.2f GiB, memory %.2f GiB per GPU. Sharding over %d devices.',
                footprint / (2 ** 30), gpu_memory / (2 ** 30), num_devices)

    user_args = [
        str(Path(__file__).parents[0] / 'diagonalize.py'),
        parameters.pkgpath,
        'cr'
    ]
    walltime = '02:00:00'
    if num_devices > 1:
        if (await HPCProfileBlock.load(parameters.hpc_profile)).hpc_target != 'local':
            user_args += ['--mpi']
            walltime = '04:00:00'
        elif (gpus := (await Variable.get(parameters.hpc_profile)).get('gpus')):
            if isinstance(gpus, list):
                gpus = ','.join(f'{i}' for i in gpus)
            user_args += ['--gpus', gpus]

    result = await run_job_from_blocks(
        command_block_name=parameters.python_cuda_cmd,
        execution_profile_block_name=parameters.python_gpu_mpi_exec_profile,
        hpc_profile_block_name=parameters.hpc_profile,
        work_dir=Path(parameters.pkgpath) / 'work/diagonalize',
        user_args=user_args,
        script_filename='run',
        execution_profile_overrides={'walltime': walltime, 'num_nodes': num_devices}
    )
    if result.exit_status != 0:
        raise RuntimeError(f"Diagonalize job failed: exit_status={result.exit_status}")

    return load_skqd_result(parameters, 'cr')[0][-1]


def run():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog='skqd_z2lgt')
    parser.add_argument('parameters', metavar='PATH',
                        help='Path to a yaml file containing the workflow parameters.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    if os.path.isdir(options.parameters):
        parameters = open_output(options.parameters)
    else:
        parameters = open_from_yaml(options.parameters, overwrite=options.overwrite)

    asyncio.run(skqd_z2lgt(parameters))


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_z2lgt.with_options(
        version=os.getenv("WF_VERSION", "unknown")
    ).serve(
        name="skqd_z2lgt",
        description="SKQD experiment for Z2 LGT."
    )
