"""Ground state of the 2D Z2 LGT by DMRG on an NVIDIA GPU (CuPy) or CPU (NumPy).

Runs the dual Ising model (as in SKQD) and/or the primal link model with Gauss-law penalty, in the
charge sector given by lgt.charged_vertices, and writes energies and diagnostics to HDF5.

Example:
    python scripts/run_dmrg_gpu.py yaml/dmrg_gpu_conf.yaml --model both -o dmrg_156q.h5 \\
        --backend cupy --device 0
"""
from argparse import ArgumentParser
from dataclasses import asdict
import json
import logging
import sys
import yaml
import numpy as np
import h5py
from skqd_z2lgt.orchestration.parameters import LGTParameters
from skqd_z2lgt.dmrg_gpu.backend import xp
from skqd_z2lgt.dmrg_gpu.models import make_lattice
from skqd_z2lgt.dmrg_gpu.runner import solve, DEFAULT_SWEEPS
from skqd_z2lgt.dmrg_gpu.sweeps import Sweeps


def main():
    parser = ArgumentParser(prog='run_dmrg_gpu.py')
    parser.add_argument('conf', metavar='PATH',
                        help='YAML file with an "lgt" section (same format as parameters.yaml) and'
                        ' an optional "dmrg_gpu" section.')
    parser.add_argument('-o', '--out', metavar='PATH', default='dmrg_gpu.h5')
    parser.add_argument('--model', choices=['dual', 'primal', 'primal-sym', 'both', 'all'],
                        default=None,
                        help='both = dual + primal; all = dual + primal + primal-sym')
    parser.add_argument('--backend', choices=['auto', 'cupy', 'numpy'], default='auto')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', choices=['single', 'double'], default='double')
    parser.add_argument('--plaquette-energy', type=float, help='Override lgt.plaquette_energy.')
    parser.add_argument('--charged-vertices', type=int, nargs='*',
                        help='Override lgt.charged_vertices (no values = vacuum).')
    # Sweep schedule overrides (ITensors-style: the last value repeats for remaining sweeps)
    parser.add_argument('--nsweeps', type=int)
    parser.add_argument('--maxdim', type=int, nargs='+')
    parser.add_argument('--mindim', type=int, nargs='+')
    parser.add_argument('--cutoff', type=float, nargs='+')
    parser.add_argument('--noise', type=float, nargs='+')
    parser.add_argument('--krylovdim', type=int, nargs='+')
    parser.add_argument('--sweeps-table', metavar='PATH',
                        help='File with an ITensors-style sweep table (header: maxdim mindim cutoff'
                        ' noise ...), replacing dmrg_gpu.sweeps.')
    parser.add_argument('--penalty', type=float)
    parser.add_argument('--svd', choices=['svd', 'eigh'])
    parser.add_argument('--ordering',
                        choices=['snake', 'snake-rows', 'snake-columns', 'rcm', 'natural'])
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--save-mps', action='store_true')
    parser.add_argument('--measure-variance', action='store_true',
                        help='Energy variance after every sweep (enables variance extrapolation).')
    parser.add_argument('--checkpoint', metavar='PREFIX',
                        help='Write a checkpoint PREFIX_<model>.npz after every sweep.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the checkpoints given by --checkpoint if they exist.')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()), stream=sys.stderr,
                        format='%(asctime)s:%(name)s:%(levelname)s %(message)s')

    with open(options.conf, 'r', encoding='utf-8') as source:
        conf = yaml.safe_load(source)
    lgt = LGTParameters(**conf.get('lgt', {}))
    if options.plaquette_energy is not None:
        lgt.plaquette_energy = options.plaquette_energy
    if options.charged_vertices is not None:
        lgt.charged_vertices = options.charged_vertices

    dconf = {
        'model': 'both',
        'sweeps': dict(DEFAULT_SWEEPS),
        'penalty': 4.,
        'svd': 'svd',
        'ordering': 'snake',
        'energy_tol': None,
        'num_samples': 0,
        'seed': 0,
    }
    dconf.update(conf.get('dmrg_gpu', {}) or {})
    for key in ['model', 'penalty', 'svd', 'ordering', 'num_samples']:
        if (value := getattr(options, key)) is not None:
            dconf[key] = value

    if options.sweeps_table:
        with open(options.sweeps_table, 'r', encoding='utf-8') as source:
            dconf['sweeps'] = {'table': source.read()}
    sweep_conf = dconf['sweeps']
    sweep_conf = {'table': sweep_conf} if isinstance(sweep_conf, str) else dict(sweep_conf)
    for key in ['nsweeps', 'maxdim', 'mindim', 'cutoff', 'noise', 'krylovdim']:
        if (value := getattr(options, key)) is not None:
            sweep_conf[key] = value
    sweeps = Sweeps.from_config(sweep_conf)
    dconf['sweeps'] = sweeps.to_dict()
    logging.info('Sweep schedule:\n%s', sweeps)

    backend = xp.configure(options.backend, device=options.device, precision=options.precision)
    logging.info('Backend: %s %s', backend, xp.memory_info())

    lattice = make_lattice(lgt.lattice)
    models = {'both': ['dual', 'primal'], 'all': ['dual', 'primal', 'primal-sym']}.get(
        dconf['model'], [dconf['model']])
    results = {}
    for model in models:
        def progress(record, model=model):
            print(json.dumps({'model': model, **asdict(record)}), flush=True)

        results[model] = solve(
            lattice, lgt.plaquette_energy, lgt.charged_vertices, model=model,
            charged_plaquettes=lgt.charged_plaquettes, penalty=dconf['penalty'],
            sweeps=sweeps, ordering=dconf['ordering'], svd_method=dconf['svd'],
            energy_tol=dconf['energy_tol'], num_samples=dconf['num_samples'],
            seed=dconf['seed'], callback=progress,
            measure_variance=options.measure_variance,
            checkpoint=f'{options.checkpoint}_{model}.npz' if options.checkpoint else None,
            resume=options.resume
        )
        res = results[model]
        summary = {'model': model, 'energy': res['energy'], 'dmrg_energy': res['dmrg_energy'],
                   'penalty_energy': res['penalty_energy'],
                   'max_mps_bond': int(res['mps_bond_dims'].max(initial=1)),
                   'max_mpo_bond': int(res['mpo_bond_dims'].max(initial=1)),
                   'variance': float(res['variance'])}
        if 'error' in res['extrapolation']:
            summary['extrapolated'] = res['extrapolation']['energy_extrapolated']
            summary['error_estimate'] = res['extrapolation']['error']
        if model.startswith('primal'):
            summary['max_gauss_violation'] = float(
                np.max(np.abs(res['gauss_expectations'] - res['gauss_targets']), initial=0.))
        print(json.dumps(summary), flush=True)

    for model in ('primal', 'primal-sym'):
        if model in results and 'dual' in results:
            diff = results[model]['energy'] - results['dual']['energy']
            print(json.dumps({f'{model}_minus_dual': diff}), flush=True)

    with h5py.File(options.out, 'w') as out:
        cgr = out.create_group('configuration')
        cgr.create_dataset('lattice', data=str(lgt.lattice))
        cgr.create_dataset('plaquette_energy', data=lgt.plaquette_energy)
        cgr.create_dataset('charged_vertices', data=np.array(lgt.charged_vertices, dtype=int))
        cgr.create_dataset('dmrg', data=json.dumps(dconf))
        cgr.create_dataset('backend', data=backend)
        for model, res in results.items():
            group = out.create_group(model)
            group.create_dataset('variance', data=res['variance'])
            group.create_dataset('extrapolation', data=json.dumps(res['extrapolation']))
            for key in ['energy', 'dmrg_energy', 'penalty_energy', 'gauss_expectations',
                        'gauss_targets', 'order', 'mpo_bond_dims', 'mps_bond_dims', 'link_state']:
                group.create_dataset(key, data=res[key])
            history = group.create_group('history')
            for field in ['sweep', 'maxdim', 'cutoff', 'noise', 'energy', 'max_bond',
                          'max_trunc_err', 'seconds', 'variance']:
                values = [getattr(rec, field) for rec in res['history']]
                values = [np.nan if v is None else v for v in values]
                history.create_dataset(field, data=values)
            if 'samples' in res:
                group.create_dataset('samples', data=np.packbits(res['samples'], axis=-1))
                group['samples'].attrs['num_bits'] = res['samples'].shape[1]
            if options.save_mps:
                mgr = group.create_group('mps')
                for site, tensor in enumerate(res['mps']):
                    mgr.create_dataset(f'{site}', data=xp.asnumpy(tensor))

    logging.info('Output written to %s', options.out)


if __name__ == '__main__':
    main()
