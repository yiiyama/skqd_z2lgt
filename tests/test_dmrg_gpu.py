"""Checks of the GPU DMRG against the MPO builder and exact diagonalization (runs on CPU too)."""
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh
from skqd_z2lgt.dmrg_gpu.backend import xp
from skqd_z2lgt.dmrg_gpu.mpo import (build_mpo, mpo_to_dense, terms_to_dense, pauli_terms,
                                     permute_terms, site_order)
from skqd_z2lgt.dmrg_gpu.models import make_lattice, base_link_state, dual_problem, primal_problem
from skqd_z2lgt.dmrg_gpu.runner import solve
from skqd_z2lgt.dmrg_gpu.sweeps import Sweeps
from skqd_z2lgt.dmrg_gpu.mpo import snake_order, cut_sizes

SWEEP_TABLE = '''
maxdim mindim cutoff noise
16     4      1e-12  1e-3
32     4      1e-12  1e-4
64     4      1e-12  1e-5
128    4      1e-12  0
128    4      1e-12  0
'''

SMALL = '''
 * * *
* * * *
 * * *
'''


@pytest.fixture(autouse=True, scope='module')
def backend():
    xp.configure('auto', precision='double')


def test_mpo_matches_dense():
    rng = np.random.default_rng(1)
    nsites = 7
    for _ in range(20):
        terms = [(rng.normal(), {})]
        for _ in range(rng.integers(1, 15)):
            sites = rng.choice(nsites, rng.integers(1, 5), replace=False)
            terms.append((rng.normal(), {int(s): str(rng.choice(['X', 'Z'])) for s in sites}))
        for compress in (False, True):
            mpo = build_mpo(terms, nsites, compress=compress)
            assert np.allclose(mpo_to_dense(mpo), terms_to_dense(terms, nsites), atol=1.e-10)


def test_primal_mpo_with_penalty_matches_dense():
    lattice = make_lattice(' *\n* *\n *')
    problem = primal_problem(lattice, 0.7, base_link_state(lattice, [0, 3]), penalty=3.)
    terms = problem.terms()
    order = site_order(terms, problem.num_qubits)
    site_terms = permute_terms(terms, order)
    mpo = build_mpo(site_terms, problem.num_qubits)
    assert np.allclose(mpo_to_dense(mpo), terms_to_dense(site_terms, problem.num_qubits))


def dual_exact(lattice, coupling, charges):
    hamiltonian = dual_problem(lattice, coupling, base_link_state(lattice, charges)).hamiltonian
    return eigsh(hamiltonian.to_matrix(sparse=True).real, k=1, which='SA')[0][0]


@pytest.mark.parametrize('charges', [[], [0, 4]])
@pytest.mark.parametrize('coupling', [0.5, 1.0, 2.0])
def test_dual_and_primal_match_exact(charges, coupling):
    lattice = make_lattice(SMALL)
    exact = dual_exact(lattice, coupling, charges)
    dual = solve(lattice, coupling, charges, model='dual',
                 sweeps={'nsweeps': 4, 'maxdim': [16, 32, 64], 'cutoff': 1.e-12})
    primal = solve(lattice, coupling, charges, model='primal', penalty=4.,
                   sweeps={'nsweeps': 5, 'maxdim': [16, 32, 64, 128], 'cutoff': 1.e-12,
                           'noise': [1.e-3, 1.e-4, 1.e-5, 0.]})
    assert abs(dual['energy'] - exact) < 1.e-8
    assert abs(primal['energy'] - exact) < 1.e-7
    assert primal['penalty_energy'] < 1.e-8
    assert np.allclose(primal['gauss_expectations'], primal['gauss_targets'], atol=1.e-6)


def test_primal_exact_with_penalty():
    """The penalized primal ground state equals the dual ground state also by brute force."""
    lattice = make_lattice(' * *\n* * *\n * *')
    charges = [0, 6]
    exact = dual_exact(lattice, 1.0, charges)
    problem = primal_problem(lattice, 1.0, base_link_state(lattice, charges), penalty=4.)
    terms = problem.terms()
    matrix = terms_to_dense(terms, problem.num_qubits)
    # terms_to_dense uses site 0 as most significant; spectrum is independent of the ordering
    assert abs(np.linalg.eigvalsh(matrix)[0] - exact) < 1.e-9


def test_eigh_truncation_and_sampling():
    lattice = make_lattice(SMALL)
    exact = dual_exact(lattice, 1.0, [0, 4])
    result = solve(lattice, 1.0, [0, 4], model='primal', svd_method='eigh',
                   sweeps=SWEEP_TABLE, num_samples=200)
    assert abs(result['energy'] - exact) < 1.e-7
    # Every sampled link configuration must satisfy Gauss's law of the charge sector
    for support, target in zip(result['gauss_supports'], result['gauss_targets']):
        parity = result["samples"][:, support].astype(int).sum(axis=1) % 2
        assert np.all(1 - 2 * parity == target)


def test_sweeps_schedule():
    sweeps = Sweeps(nsweeps=5, maxdim=[10, 20, 100], cutoff=1.e-8, noise=[1.e-5, 0.])
    assert [p.maxdim for p in sweeps] == [10, 20, 100, 100, 100]
    assert [p.noise for p in sweeps] == [1.e-5, 0., 0., 0., 0.]
    assert all(p.cutoff == 1.e-8 for p in sweeps)

    table = Sweeps.from_config(SWEEP_TABLE)
    assert len(table) == 5 and table[2].maxdim == 64 and table[3].noise == 0. and table[0].mindim == 4
    longer = Sweeps.from_config({'table': SWEEP_TABLE, 'nsweeps': 7})
    assert longer[6].maxdim == 128
    assert Sweeps.from_config({'maxdim': [8, 16, 32]}).nsweeps == 3


def test_snake_ordering():
    lattice = make_lattice((4, 7))
    problem = dual_problem(lattice, 1.0, base_link_state(lattice, [9, 13]))
    order = snake_order(problem.positions, 'rows')
    assert sorted(order) == list(range(problem.num_qubits))
    # Consecutive rows are traversed in opposite directions
    rows = problem.positions[order, 1]
    first_row = order[rows == rows.min()]
    assert np.all(np.diff(problem.positions[first_row, 0]) > 0)
    terms = problem.terms()
    snake = site_order(terms, problem.num_qubits, 'snake', positions=problem.positions)
    assert cut_sizes(terms, snake).max() <= cut_sizes(terms, site_order(terms, problem.num_qubits,
                                                                         'rcm')).max()


@pytest.mark.parametrize('ordering', ['snake', 'rcm', 'natural'])
def test_orderings_give_same_energy(ordering):
    lattice = make_lattice(SMALL)
    exact = dual_exact(lattice, 1.0, [0, 4])
    for model in ['dual', 'primal']:
        result = solve(lattice, 1.0, [0, 4], model=model, ordering=ordering, sweeps=SWEEP_TABLE)
        assert abs(result['energy'] - exact) < 1.e-7


def test_primal_sym_exact_gauss_law():
    lattice = make_lattice(SMALL)
    for charges in ([], [0, 4]):
        for coupling in (0.5, 2.0):
            exact = dual_exact(lattice, coupling, charges)
            result = solve(lattice, coupling, charges, model='primal-sym', sweeps=SWEEP_TABLE)
            assert abs(result['energy'] - exact) < 1.e-8
            # Gauss's law holds exactly, not just approximately
            assert np.max(np.abs(result['gauss_expectations'] - result['gauss_targets'])) < 1.e-12
            assert result['variance'] < 1.e-7


def test_variance_and_extrapolation():
    lattice = make_lattice((4, 5))
    exact = dual_exact(lattice, 1.5, [7, 10])
    converged = solve(lattice, 1.5, [7, 10], model='dual', sweeps=SWEEP_TABLE)
    assert abs(converged['energy'] - exact) < 1.e-9 and converged['variance'] < 1.e-8
    sweeps = {'nsweeps': 12, 'maxdim': [8, 8, 4, 4, 5, 5, 6, 6, 8, 8], 'cutoff': 1.e-14,
              'noise': [1.e-4, 1.e-5, 0.]}
    truncated = solve(lattice, 1.5, [7, 10], model='dual', sweeps=sweeps, measure_variance=True)
    extrap = truncated['extrapolation']
    assert extrap['npoints'] >= 2 and 'variance' in extrap
    error = extrap['energy_best'] - exact
    assert error > 0.
    # The reported error bar must cover the actual error of the best variational energy
    assert extrap['error'] >= 0.5 * error


@pytest.mark.parametrize('model', ['dual', 'primal', 'primal-sym'])
def test_checkpoint_resume(tmp_path, model):
    lattice = make_lattice(SMALL)
    exact = dual_exact(lattice, 1.0, [0, 4])
    path = str(tmp_path / 'ckpt.npz')
    schedule = Sweeps.from_config(SWEEP_TABLE)
    partial = Sweeps(nsweeps=2, maxdim=schedule.maxdim[:2], mindim=schedule.mindim[:2],
                     cutoff=schedule.cutoff[:2], noise=schedule.noise[:2])
    first = solve(lattice, 1.0, [0, 4], model=model, sweeps=partial, checkpoint=path)
    assert len(first['history']) == 2
    resumed = solve(lattice, 1.0, [0, 4], model=model, sweeps=schedule, checkpoint=path,
                    resume=True)
    assert [rec.sweep for rec in resumed['history']] == list(range(len(schedule)))
    assert abs(resumed['energy'] - exact) < 1.e-7
    with pytest.raises(ValueError):
        solve(lattice, 1.1, [0, 4], model=model, sweeps=schedule, checkpoint=path, resume=True)


@pytest.mark.parametrize('model', ['dual', 'primal-sym'])
def test_warm_start_scan(model):
    from skqd_z2lgt.dmrg_gpu.scan import scan
    lattice = make_lattice(SMALL)
    couplings = [0.5, 1.0, 1.5, 2.0]
    results = scan(lattice, couplings, [0, 4], model=model, sweeps=SWEEP_TABLE,
                   warm_sweeps={'nsweeps': 4, 'maxdim': [128], 'cutoff': 1.e-12,
                                'noise': [1.e-4, 1.e-5, 0.]},
                   cold_every=None, bidirectional=True)
    for res, coupling in zip(results, couplings):
        assert abs(res['energy'] - dual_exact(lattice, coupling, [0, 4])) < 1.e-7
        assert not res['flags']
    assert 'warm-fwd' in results[1]['energies'] and 'warm-bwd' in results[1]['energies']
