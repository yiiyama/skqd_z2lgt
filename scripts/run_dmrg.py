"""Run DMRG."""
from argparse import ArgumentParser
import yaml
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ising_dmrg import ising_dmrg

JULIA_BIN = ['julia', '--sysimage', '/opt/julia/iiyama/sysimages/sys_itensors.so']


if __name__ == '__main__':
    parser = ArgumentParser(prog='run_dmrg.py')
    parser.add_argument('conf', metavar='PATH',
                        help='Path to a yaml file containing the simulation configuration.')

    options = parser.parse_args()
    with open(options.conf, 'r', encoding='utf-8') as source:
        conf = yaml.load(source, yaml.Loader)

    dual_lattice = TriangularZ2Lattice(conf['lattice_config']).plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(conf['coupling'])
    if conf.get('invert', False):
        hamiltonian *= -1.

    print(f'Hamiltonian parameters: Np={hamiltonian.num_qubits}, coupling={conf["coupling"]}')

    ising_dmrg(hamiltonian, conf['filename'], JULIA_BIN)
