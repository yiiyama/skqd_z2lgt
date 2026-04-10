"""Compute the ground state of a large system through a distributed eigensolver."""
from argparse import ArgumentParser
from dataclasses import dataclass
import logging
import yaml
import numpy as np
from scipy.sparse.linalg import eigsh
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ising_hamiltonian import parse_hamiltonian, make_matvec, ising_hamiltonian
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard


@dataclass
class Configuration:
    """Simulation configuration."""
    lattice_config: str
    coupling: float

    @classmethod
    def from_dict(cls, conf_dict):
        return cls(**conf_dict)

    def save(self, fd: h5py.File):
        gr = fd.create_group('configuration')
        gr.create_dataset('lattice', data=self.lattice_config)
        gr.create_dataset('coupling', data=self.coupling)


if __name__ == '__main__':
    parser = ArgumentParser(prog='compute_ground_state.py')
    parser.add_argument('conf', metavar='PATH',
                        help='Path to a yaml file containing the Hamiltonian parameters.')
    parser.add_argument('-o', '--out', metavar='PATH', default='out.h5',
                        help='Output file path.')
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help='Logging level.')
    options = parser.parse_args()

    log_level = getattr(logging, options.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    # Construct the configuration object
    with open(options.conf, 'r', encoding='utf-8') as source:
        conf = Configuration.from_dict(yaml.load(source, yaml.Loader))

    logging.info('Calculating ground state for Hamiltonian with parameters:\n'
                 'Lattice:\n%s\ncoupling: %.2f',
                 conf.lattice_config.strip('\n'), conf.coupling)

    # Set up the lattice
    dual_lattice = TriangularZ2Lattice(conf.lattice_config).plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(conf.coupling)

    isingh = ising_hamiltonian(hamiltonian, npmod=jnp)
    logging.info('Compiling matvec..')
    isingh.matvec(np.zeros(2 ** hamiltonian.num_qubits, dtype=np.complex128))
    logging.info('Computing the ground state')
    eigvals, eigvecs = eigsh(isingh, k=1, which='SA')
    # nq, zzops, zops, xops = parse_hamiltonian(hamiltonian)
    # matvec = make_matvec(nq, zzops, zops, xops, jnp)
    # mesh = jax.make_mesh((jax.device_count(), 1), ('dev', 'dum'))
    # xmat = jax.device_put(np.full((2 ** nq, 1), np.power(2., -nq / 2), dtype=np.complex128),
    #                       NamedSharding(mesh, PartitionSpec('dev', 'dum')))
    # logging.info('Placed a %d-qubit vector on %d devices. Diagonalizing', nq, jax.device_count())
    # # pylint: disable-next=unbalanced-tuple-unpacking
    # eigvals, eigvecs, _ = lobpcg_standard(matvec, xmat)

    with h5py.File(options.out, 'w') as output:
        conf.save(output)
        output.create_dataset('energy', data=eigvals[0])
        output.create_dataset('state', data=eigvecs[:, 0])

    logging.info('Output written at %s. Normal exit.', options.out)
