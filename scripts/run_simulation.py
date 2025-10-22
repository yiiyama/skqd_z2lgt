"""Run a noisy simulation."""
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
import logging
import pickle
import json
import time
from typing import Any, Optional
import yaml
import numpy as np
import h5py
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.utils import as_bitarray
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits
from skqd_z2lgt.noise_model import subset_noise_model


@dataclass
class Configuration:
    """Simulation configuration."""
    lattice_config: str
    plaquette_energy: float
    delta_t: float
    steps: list[int]
    basis_2q: str
    num_shots: int
    simulator_options: dict[str, Any] = field(default_factory=dict)
    instance: Optional[str] = None
    backend_name: Optional[str] = None
    noise_model: Optional[NoiseModel] = None

    def __post_init__(self):
        if isinstance(self.steps, int):
            self.steps = list(range(1, self.steps + 1))

        if isinstance(self.noise_model, str):
            with open(self.noise_model, 'rb') as src:
                self.noise_model = pickle.load(src)[2]

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
        gr.create_dataset('simulator_options', data=json.dumps(self.simulator_options))
        if self.instance:
            gr.create_dataset('instance', data=self.instance)
            gr.create_dataset('backend_name', data=self.backend_name)
        if self.noise_model:
            gr.create_dataset('noise_model', data=pickle.dumps(self.noise_model))


if __name__ == '__main__':
    parser = ArgumentParser(prog='run_simulation.py')
    parser.add_argument('conf', metavar='PATH',
                        help='Path to a yaml file containing the simulation configuration.')
    parser.add_argument('-o', '--out', metavar='PATH', default='sim_result.h5',
                        help='Output file path.')
    parser.add_argument('--save-noise-model', metavar='PATH', help='Save the noise model and exit.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    log_level = getattr(logging, options.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s:%(name)s:%(levelname)s %(message)s')
    logging.getLogger('qiskit_aer').setLevel(logging.DEBUG)

    # Construct the configuration object
    with open(options.conf, 'r', encoding='utf-8') as source:
        conf = Configuration.from_dict(yaml.load(source, yaml.Loader))

    logging.info('Setting up a Z2 LGT simulation with parameters:\n'
                 'Lattice:\n%s\nplaquette_energy: %.2f\ndelta_t: %.2f\nsteps: %s',
                 conf.lattice_config.strip('\n'), conf.plaquette_energy, conf.delta_t, conf.steps)

    # Set up the lattice
    lattice = TriangularZ2Lattice(conf.lattice_config)

    # Obtain the layout and subset noise model
    if conf.noise_model is None:
        logging.info('Retriving data for instance "%s" backend %s..',
                     conf.instance, conf.backend_name)
        service = QiskitRuntimeService(instance=conf.instance)
        backend = service.backend(conf.backend_name)

        full_noise_model = NoiseModel.from_backend(backend)

        layout = lattice.layout_heavy_hex(backend.coupling_map,
                                          backend_properties=backend.properties(),
                                          basis_2q=conf.basis_2q)
        logging.info('Selected qubits: %s', layout)
        noise_model = subset_noise_model(full_noise_model, layout)

        if options.save_noise_model:
            with open(options.save_noise_model, 'wb') as out:
                pickle.dump((conf.backend_name, layout, noise_model), out)
            sys.exit(0)
    else:
        noise_model = conf.noise_model

    # Set up the circuits
    step_circuits_log = make_step_circuits(lattice, conf.plaquette_energy, conf.delta_t,
                                           conf.basis_2q)
    logging.info('Single Trotter step (%d qubits) gate counts: %s',
                 step_circuits_log[0].num_qubits, step_circuits_log[0].count_ops())
    start = time.time()
    step_circuits = transpile(step_circuits_log, basis_gates=['x', 'sx', 'rz', conf.basis_2q],
                              initial_layout=list(range(lattice.num_qubits)), optimization_level=2)
    logging.info('Circuit transpilation took %.2f seconds.', time.time() - start)
    single_trotter, forward, backward, measure = step_circuits
    single_id = forward.compose(backward)

    # trotter_circuits = compose_trotter_circuits(single_trotter, measure, conf.steps)
    id_circuits = compose_trotter_circuits(single_id, measure, conf.steps)

    # Run the simulator
    simulator = AerSimulator(noise_model=noise_model, **conf.simulator_options)
    logging.info('Simulating %d circuits for %d shots', len(id_circuits), conf.num_shots)
    logging.info('Simulator options: %s', conf.simulator_options)
    start = time.time()
    sim_job = simulator.run(id_circuits, shots=conf.num_shots)
    sim_result = sim_job.result()
    logging.info('Simulation took %.2f seconds.', time.time() - start)
    logging.info('%s', sim_result.metadata)
    sim_result_counts = [sim_result.get_counts(icirc) for icirc in range(len(id_circuits))]

    with h5py.File(options.out, 'w') as output:
        conf.save(output)
        for icirc, istep in enumerate(conf.steps):
            group = output.create_group(f'id_step{istep}')
            counts = sim_result_counts[icirc]
            link_states = np.array([as_bitarray(link_state) for link_state in counts.keys()])
            group.create_dataset('link_states', data=link_states)
            group.create_dataset('counts', data=list(counts.values()))

    logging.info('Output written at %s. Normal exit.', options.out)
