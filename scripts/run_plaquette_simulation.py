"""Run a noisy simulation."""
from argparse import ArgumentParser
from dataclasses import dataclass, field
import logging
import json
import time
from typing import Any
import yaml
import numpy as np
import h5py
from qiskit_aer import AerSimulator
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.plaquette_dual import PlaquetteDual
from heavyhex_qft.utils import as_bitarray
from skqd_z2lgt.circuits import make_plaquette_circuits, compose_trotter_circuits


@dataclass
class Configuration:
    """Simulation configuration."""
    lattice_config: str
    plaquette_energy: float
    delta_t: float
    steps: list[int]
    num_shots: int
    simulator_options: dict[str, Any] = field(default_factory=dict)

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
        gr.create_dataset('num_shots', data=self.num_shots)
        gr.create_dataset('simulator_options', data=json.dumps(self.simulator_options))


if __name__ == '__main__':
    parser = ArgumentParser(prog='run_plaquette_simulation.py')
    parser.add_argument('conf', metavar='PATH',
                        help='Path to a yaml file containing the simulation configuration.')
    parser.add_argument('-o', '--out', metavar='PATH', default='sim_result.h5',
                        help='Output file path.')
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
    dual_lattice = PlaquetteDual(lattice)

    # Set up the circuits
    step_circuits = make_plaquette_circuits(dual_lattice, conf.plaquette_energy, conf.delta_t)
    single_trotter = step_circuits[0]
    measure = step_circuits[-1]
    logging.info('Single Trotter step (%d qubits) gate counts: %s',
                 single_trotter.num_qubits, single_trotter.count_ops())
    trotter_circuits = compose_trotter_circuits(single_trotter, measure, conf.steps)

    # Run the simulator
    simulator = AerSimulator(**conf.simulator_options)
    logging.info('Simulating %d circuits for %d shots', len(trotter_circuits), conf.num_shots)
    logging.info('Simulator options: %s', conf.simulator_options)
    start = time.time()
    sim_job = simulator.run(trotter_circuits, shots=conf.num_shots)
    sim_result = sim_job.result()
    logging.info('Simulation took %.2f seconds.', time.time() - start)
    logging.info('%s', sim_result.metadata)
    sim_result_counts = [sim_result.get_counts(icirc) for icirc in range(len(trotter_circuits))]

    with h5py.File(options.out, 'w') as output:
        conf.save(output)
        for icirc, istep in enumerate(conf.steps):
            group = output.create_group(f'fwd_step{istep}')
            counts = sim_result_counts[icirc]
            plaq_states = np.array([as_bitarray(link_state) for link_state in counts.keys()])
            group.create_dataset('plaq_states', data=plaq_states)
            group.create_dataset('counts', data=list(counts.values()))

    logging.info('Output written at %s. Normal exit.', options.out)
