"""Run a noisy simulation."""
from argparse import ArgumentParser
from dataclasses import dataclass
import logging
import time
from typing import Optional
import yaml
import numpy as np
import h5py
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.utils import as_bitarray
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits


@dataclass
class Configuration:
    """Simulation configuration."""
    lattice_config: str
    plaquette_energy: float
    delta_t: float
    steps: list[int]
    basis_2q: str
    num_shots: int
    instance: str
    backend_name: str
    job_id: Optional[str] = None
    layout: Optional[list[int]] = None

    def __post_init__(self):
        if isinstance(self.steps, int):
            self.steps = list(range(1, self.steps + 1))

    @classmethod
    def from_dict(cls, conf_dict):
        return cls(**conf_dict)

    def save(self, fd: h5py.File):
        gr = fd.create_group('configuration')
        gr.create_dataset('lattice', data=self.lattice_config)
        gr.create_dataset('plaquette_energy', data=self.plaquette_energy)
        gr.create_dataset('delta_t', data=self.delta_t)
        gr.create_dataset('steps', data=self.steps)
        gr.create_dataset('basis_2q', data=self.basis_2q)
        gr.create_dataset('num_shots', data=self.num_shots)
        gr.create_dataset('instance', data=self.instance)
        gr.create_dataset('backend_name', data=self.backend_name)
        if self.job_id:
            gr.create_dataset('job_id', data=self.job_id)
        if self.layout:
            gr.create_dataset('layout', data=self.layout)


if __name__ == '__main__':
    parser = ArgumentParser(prog='run_experiment.py')
    parser.add_argument('conf', metavar='PATH',
                        help='Path to a yaml file containing the simulation configuration.')
    parser.add_argument('-o', '--out', metavar='PATH', default='result.h5',
                        help='Output file path.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    log_level = getattr(logging, options.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    # Construct the configuration object
    with open(options.conf, 'r', encoding='utf-8') as source:
        conf = Configuration.from_dict(yaml.load(source, yaml.Loader))

    logging.info('Setting up a Z2 LGT experiment with parameters:\n'
                 'Lattice:\n%s\nplaquette_energy: %.2f\ndelta_t: %.2f\nsteps: %s',
                 conf.lattice_config.strip('\n'), conf.plaquette_energy, conf.delta_t, conf.steps)

    # Set up the lattice
    lattice = TriangularZ2Lattice(conf.lattice_config)

    # Obtain the backend and layout
    logging.info('Retriving data for instance "%s" backend %s..', conf.instance, conf.backend_name)
    service = QiskitRuntimeService(instance=conf.instance)
    backend = service.backend(conf.backend_name)

    conf.layout = lattice.layout_heavy_hex(backend.coupling_map,
                                           backend_properties=backend.properties(),
                                           basis_2q=conf.basis_2q)
    logging.info('Selected qubits: %s', conf.layout)

    # Set up the circuits
    step_circuits_log = make_step_circuits(lattice, conf.plaquette_energy, conf.delta_t,
                                           conf.basis_2q)
    logging.info('Single Trotter step (%d qubits) gate counts: %s',
                 step_circuits_log[0].num_qubits, step_circuits_log[0].count_ops())
    start = time.time()
    step_circuits = transpile(step_circuits_log, backend=backend, initial_layout=conf.layout,
                              optimization_level=2)
    logging.info('Circuit transpilation took %.2f seconds.', time.time() - start)
    single_trotter, forward, backward, measure = step_circuits
    single_id = forward.compose(backward)

    id_circuits = compose_trotter_circuits(single_id, measure, conf.steps)
    trotter_circuits = compose_trotter_circuits(single_trotter, measure, conf.steps)
    circuits = id_circuits + trotter_circuits

    # Submit the job
    logging.info('Submitting %d circuits with %d shots each', len(circuits), conf.num_shots)
    start = time.time()
    if conf.job_id:
        job = service.job(conf.job_id)
    else:
        sampler = Sampler(backend)
        job = sampler.run(circuits, shots=conf.num_shots)
        logging.info('Job id %s. Blocking until results are ready..', job.job_id())
    result = job.result()
    logging.info('%s', result.metadata)

    group_names = [f'id_step{i}' for i in conf.steps] + [f'fwd_step{i}' for i in conf.steps]
    with h5py.File(options.out, 'w') as output:
        conf.save(output)
        for res, name in zip(result, group_names):
            group = output.create_group(name)
            counts = res.data.c.get_counts()
            link_states = np.array([as_bitarray(link_state) for link_state in counts.keys()])
            group.create_dataset('link_states', data=link_states)
            group.create_dataset('counts', data=list(counts.values()))

    logging.info('Output written at %s. Normal exit.', options.out)
