"""Definition of the main Prefect flow for Z2LGT SKQD."""
import os
from pathlib import Path
import logging
import asyncio
import tempfile
import math
import json
from concurrent.futures import ThreadPoolExecutor
from prefect import flow, task, get_run_logger
from prefect.client.schemas.filters import (ArtifactFilter, ArtifactFilterKey,
                                            ArtifactFilterTaskRunId)
from prefect.client.orchestration import get_client
from prefect.variables import Variable
from prefect.runtime import task_run
from prefect_qiskit.runtime import QuantumRuntime
from prefect_qiskit.primitives import PrimitiveJobRun
from prefect_miyabi import MiyabiJobBlock
from skqd_z2lgt.orchestration.common import make_lattice
from skqd_z2lgt.orchestration.parameters import Parameters
from skqd_z2lgt.orchestration.open_output import open_output, open_from_yaml
from skqd_z2lgt.orchestration.sample_quantum import get_trotter_circuits, compile_pubs
from skqd_z2lgt.orchestration.io import load_dmrg, load_skqd_result
from skqd_z2lgt.orchestration.flows import (
    dmrg_flow,
    sample_quantum_flow,
    preprocess_flow,
    train_generator_flow,
    diagonalize_flow
)

SCRIPT_DIR = Path(__file__).parents[1] / '_scripts'


@flow
async def skqd_z2lgt(
    parameters: Parameters,
    runtime_name: str = 'ibm-runner',
    cpujob_block_name: str = 'cpu-script',
    cudajob_block_name: str = 'cuda-script'
) -> float:
    """Calculation of ground-state energy of Z2 LGT using SKQD.

    Args:
        parameters: Configuration parameters.
        runtime_name: Name of QuantumRunner block.
        cpujob_block_name: Name of the MiyabiJobBlock that executes a python script.
        cudajob_block_name: Name of the MiyabiJobBlock that executes a python script in a CUDA
            environment.
    """
    logger = get_run_logger()
    logger.setLevel(logging.INFO)

    tmpdir = None
    if not parameters.pkgpath:
        tmpdir = tempfile.TemporaryDirectory()
        parameters.pkgpath = tmpdir.name

    open_output(parameters, logger)
    if parameters.dmrg:
        dmrg_future = dmrg.submit(parameters, cpujob_block_name=cpujob_block_name)
    await sample_quantum(parameters, runtime_name=runtime_name)
    await preprocess(parameters, cpujob_block_name=cpujob_block_name)
    await train_generator(parameters, cudajob_block_name=cudajob_block_name)
    energy_cr = await diagonalize(parameters, 'cr', cudajob_block_name=cudajob_block_name)

    if tmpdir:
        tmpdir.cleanup()

    if parameters.dmrg:
        dmrg_energy = dmrg_future.result()
        logger.info('DMRG energy: %f', dmrg_energy)
    # logger.info('SKQD energy (random bit flips): %f', energy_rn)
    logger.info('SKQD energy (full conf. recovery): %f', energy_cr)

    return energy_cr


@task
async def dmrg(
    parameters: Parameters,
    cpujob_block_name: str
) -> float:
    """Run DMRG and MPS sampling."""
    logger = get_run_logger()
    logger.info('Estimating ground-state energy via DMRG')

    async def dmrg_miyabi():
        job_block = await MiyabiJobBlock.load(cpujob_block_name)
        job_block.num_nodes = 1
        job_block.mpiprocs = 1
        job_block.walltime = '00:30:00'

        with job_block.get_executor() as executor:
            arguments = [
                SCRIPT_DIR / 'dmrg.py',
                parameters.pkgpath
            ]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job dmrg.py failed')

    def dmrg_fn():
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(dmrg_miyabi())).result()
        return load_dmrg(parameters)

    return dmrg_flow(parameters, dmrg_fn, logger)


@task
async def sample_quantum(
    parameters: Parameters,
    runtime_name: str = 'ibm-runner'
):
    """Run the circuits on a backend and return the sampler results.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRunner block.
        option_name: Name of the Variable storing sampler primitive options.
    """
    logger = get_run_logger()
    logger.info('Sampling Trotter circuit final state bitstrings')

    runtime = await QuantumRuntime.load(runtime_name)
    if parameters.runtime.backend is not None:
        runtime.resource_name = parameters.runtime.backend
    if parameters.runtime.instance is not None:
        runtime.credentials.crn = parameters.runtime.instance

    def fetch_result_fn():
        def fn():
            job = PrimitiveJobRun(job_id=parameters.runtime.job_id, credentials=runtime.credentials)
            return asyncio.run(job.fetch_result())

        with ThreadPoolExecutor(1) as executor:
            return executor.submit(fn).result()

    # Cannot call runtime.get_target and runtime.sampler directly because of async_dispatch
    # (the functions would be called in a thread running an event loop and will therefore return
    # coroutines)
    def sample_quantum_fn():
        with ThreadPoolExecutor(1) as executor:
            target = executor.submit(lambda: runtime.get_target()).result()

        layout, exp_circuits, ref_circuits = get_trotter_circuits(parameters, target, logger)
        # Run primitive
        logger.info('Submitting a runtime job')
        pubs = compile_pubs(parameters, exp_circuits, ref_circuits)

        if parameters.runtime.options is not None:
            options = Variable.get(parameters.runtime.options)

        with ThreadPoolExecutor(1) as executor:
            pub_result = executor.submit(
                lambda: runtime.sampler(sampler_pubs=pubs, options=options)
            ).result()

        prefect_client = get_client(sync_client=True)
        artifacts = prefect_client.read_artifacts(
            limit=1,
            artifact_filter=ArtifactFilter(
                key=ArtifactFilterKey(any_=['job-metrics']),
                task_run_id=ArtifactFilterTaskRunId(any_=[task_run.id])
            )
        )
        if artifacts:
            keys, values = json.loads(artifacts[0].data)
            job_id = values[keys.index('job_id')]
        else:
            job_id = ''

        return layout, pub_result, job_id

    sample_quantum_flow(parameters, fetch_result_fn, sample_quantum_fn, logger)


@task
async def preprocess(
    parameters: Parameters,
    cpujob_block_name: str
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpujob_block_name: Name of the MiyabiJobBlock that executes a python script.
    """
    logger = get_run_logger()
    logger.info('Correcting and converting link states to plaquette states')

    async def convert_miyabi(task_specs):
        job_block = await MiyabiJobBlock.load(cpujob_block_name)
        job_block.num_nodes = len(task_specs)
        job_block.mpiprocs = 1
        job_block.walltime = '00:10:00'

        with job_block.get_executor() as executor:
            arguments = [
                SCRIPT_DIR / 'preprocess.py',
                parameters.pkgpath,
                '--mpi',
                '--etype', ','.join(task[0] for task in task_specs),
                '--idt', ','.join(str(task[1]) for task in task_specs),
                '--step', ','.join(str(task[2]) for task in task_specs)
            ]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError('PBS job preprocess.py failed')

    def convert_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(convert_miyabi(task_specs))).result()

    preprocess_flow(parameters, convert_fn, logger=logger)


@task
async def train_generator(
    parameters: Parameters,
    cudajob_block_name: str
):
    """Train a CRBM per Trotter step.

    Args:
        parameters: Configuration parameters.
        cudajob_block_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
        pkgpath: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    logger = get_run_logger()
    logger.info('Training conditional restricted Boltzmann machines')

    async def run_train_job(task_specs):
        job_block = await MiyabiJobBlock.load(cudajob_block_name)
        job_block.mpiprocs = 1
        job_block.num_nodes = 1
        job_block.walltime = '01:00:00'

        async def submit(idt, step):
            with job_block.get_executor() as executor:
                arguments = [
                    SCRIPT_DIR / 'train_generator.py',
                    parameters.pkgpath,
                    '--idt', f'{idt}',
                    '--step', f'{step}'
                ]
                return await executor.execute_job(
                    arguments=arguments,
                    **job_block.get_job_variables()
                )

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for idt, step in task_specs:
                tasks.append(tg.create_task(submit(idt, step)))

        if any(task.result() != 0 for task in tasks):
            raise RuntimeError('PBS job train_generator.py failed')

    def train_fn(task_specs):
        with ThreadPoolExecutor(1) as executor:
            return executor.submit(lambda: asyncio.run(run_train_job(task_specs))).result()

    train_generator_flow(parameters, train_fn, logger=logger)


@task
async def diagonalize(
    parameters: Parameters,
    mode: str,
    cudajob_block_name: str
) -> float:
    """Perform SQD with iterative configuration recovery.

    Args:
        parameters: Configuration parameters.
        mode: 'cr' or 'rn'.
        cudajob_block_name: Name of the MiyabiJobBlock that executes the python interpreter in a
            CUDA environment.
    """
    logger = get_run_logger()
    logger.info('Performing subspace diagonalization with mode %s', mode)

    # Determine the required number of nodes
    gen_size = (len(parameters.circuit.dts) * len(parameters.circuit.sampled_steps)
                * parameters.runtime.shots_exp * parameters.skqd.num_gen)
    num_p = make_lattice(parameters).num_active_plaquettes
    pt_core_size = sum(math.comb(num_p, i) for i in range(parameters.skqd.pt_core_max + 1))
    xcache_size_gib = 4 * (gen_size + pt_core_size) * (num_p + 1) / (2 ** 30)
    num_nodes = 2 ** int(math.ceil(math.log2(math.ceil(xcache_size_gib / 120. * 4))))

    async def run_diagonalize_job():
        job_block = await MiyabiJobBlock.load(cudajob_block_name)
        job_block.mpiprocs = 1
        job_block.num_nodes = num_nodes
        job_block.walltime = '24:00:00'

        with job_block.get_executor() as executor:
            arguments = [
                SCRIPT_DIR / 'diagonalize.py',
                parameters.pkgpath,
                mode,
                '--mpi'
            ]
            exit_status = await executor.execute_job(
                arguments=arguments,
                **job_block.get_job_variables()
            )
        if exit_status != 0:
            raise RuntimeError(f'PBS job diagonalize.py {mode} failed')

    def diagonalize_fn():
        with ThreadPoolExecutor(1) as executor:
            executor.submit(lambda: asyncio.run(run_diagonalize_job())).result()
        return float(load_skqd_result(parameters, mode)[0][-1])

    return diagonalize_flow(parameters, mode, diagonalize_fn, logger=logger)


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(Path(__file__).parent)

    # Deploy the workflow with specified options.
    skqd_z2lgt.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="skqd_z2lgt",
        description="SKQD experiment for Z2 LGT."
    )


if __name__ == '__main__':
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
