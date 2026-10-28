# DMRG for the 2D Z2 lattice gauge theory (`skqd_z2lgt.dmrg_gpu`)

Method and implementation notes: [docs/dmrg_method.md](../../docs/dmrg_method.md).

Classical reference energies for the SKQD workflow: ground state (and optionally low excited
states) of the pure Z2 LGT on the triangular lattice, in the vacuum or a static-charge sector,
computed by two-site DMRG on CPU (NumPy) or NVIDIA GPU (CuPy).

It replaces the ITensors.jl path in `skqd_z2lgt/ising_dmrg.py` and needs no Julia.

## What is computed

The LGT Hamiltonian, with one qubit per link, is

```
H = - sum_l Z_l - K sum_p X_l1 X_l2 X_l3          (K = lgt.plaquette_energy = lambda)
```

with Gauss-law operators `G_v = prod_{l in v} Z_l`. A charge sector is a list of charged vertices
(`q_v = -1` there, `+1` elsewhere). The sector is fixed through the minimum-weight base link state
`b` (`skqd_z2lgt.utils.minimum_weight_link_state`), exactly as in the SKQD workflow.

The same ground state is computed in three formulations:

| model | sites | Hamiltonian | how the charge sector enters |
|---|---|---|---|
| `dual` | active plaquettes | `lattice.plaquette_dual(b).make_hamiltonian(K)`: `-K X_p` plus `s_l Z_p Z_p'` (`s_l Z_p` on boundary links) | signs `s_l = -1 + 2 b_l` of the couplings along the string |
| `primal` | links | `H + penalty * sum_v (1 - q_v G_v) / 2` | penalty on violated Gauss laws; DMRG starts from `|b>` |
| `primal-sym` | links | `H` | exact Gauss-law symmetry: block-sparse MPS whose bond labels carry the vertex parities; the state never leaves the sector |

The `dual` model is exactly the Hamiltonian that SKQD diagonalizes. On an open planar lattice all
formulations have the same spectrum within the sector, so their energies must agree. The penalty is
equivalent to `(penalty/4) * sum_v (G_v - q_v)^2`, because `G_v^2 = 1`. Each violated vertex costs
`penalty`.

Reported energies are always the physical `<H>`, with the penalty excluded.

## Installation

```bash
uv pip install -e .                  # CPU only
uv pip install -e '.[gpu-cuda12]'    # + CuPy for CUDA 12 (use gpu-cuda11 for CUDA 11)
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

This requires a `heavyhex-qft` recent enough to provide `plaquette_dual` and the `(nrow, ncol)`
lattice constructor.

## Command line

```bash
# Charged sector [0, 4] on the full 156q lattice, both models
python scripts/run_dmrg_gpu.py yaml/dmrg_gpu_conf.yaml -o q0_4.h5

# Vacuum (no values after --charged-vertices), dual model only, CPU
python scripts/run_dmrg_gpu.py yaml/dmrg_gpu_conf.yaml --charged-vertices --model dual \
    --backend numpy -o vac.h5

# All three models, energy variance per sweep, checkpoints (resumable with --resume)
python scripts/run_dmrg_gpu.py yaml/dmrg_gpu_conf.yaml --model all --measure-variance \
    --checkpoint ckpt/q0_4 -o q0_4_all.h5

# Other coupling, explicit sweep schedule, GPU 1
python scripts/run_dmrg_gpu.py yaml/dmrg_gpu_conf.yaml --plaquette-energy 1.2 --backend cupy \
    --device 1 --nsweeps 12 --maxdim 64 256 1024 --noise 1e-4 1e-6 0 -o k1.2.h5
```

The configuration file has the same `lgt` section as `yaml/parameters.yaml` (`lattice`,
`plaquette_energy`, `charged_vertices`, `charged_plaquettes`), plus a `dmrg_gpu` section. Omitting
`lattice` selects `full_156q`.

| `dmrg_gpu` key | default | meaning |
|---|---|---|
| `model` | `both` | `dual`, `primal`, `primal-sym`, `both` (dual + primal) or `all` |
| `sweeps` | see below | sweep schedule |
| `penalty` | 4 | Gauss-law penalty per violated vertex (primal) |
| `svd` | `svd` | truncation by `svd` (gesvd) or `eigh` of the reduced density matrix (faster on GPU at large bond dimension) |
| `ordering` | `snake` | qubit-to-chain mapping: geometric boustrophedon `snake` (direction chosen automatically), `snake-rows`, `snake-columns`, reverse Cuthill-McKee `rcm`, or `natural` |
| `energy_tol` | none | early stop on energy change (see below) |
| `num_samples` | 0 | number of bit strings sampled from the final MPS |

Command-line options override the file: `--model`, `--backend {auto,cupy,numpy}`, `--device`,
`--precision {double,single}`, `--plaquette-energy`, `--charged-vertices`, `--penalty`, `--svd`,
`--ordering`, `--num-samples`, `--save-mps`, `--measure-variance`, `--checkpoint PREFIX`
(writes `PREFIX_<model>.npz` after every sweep), `--resume`, plus the sweep options below.

### Sweep schedule (ITensors-style)

The schedule follows ITensors' `Sweeps`. One sweep is a left-to-right pass followed by a
right-to-left pass. Each parameter is a per-sweep list, and a list shorter than `nsweeps` repeats
its last value.

| parameter | meaning |
|---|---|
| `maxdim` | maximum bond dimension |
| `mindim` | minimum bond dimension kept regardless of `cutoff` |
| `cutoff` | maximum discarded weight per truncation |
| `noise` | White's density-matrix perturbation |
| `krylovdim` | Lanczos Krylov dimension of the local eigensolver |
| `maxiter` | Lanczos restarts |

As lists:

```yaml
dmrg_gpu:
  sweeps:
    nsweeps: 14
    maxdim: [32, 64, 128, 256, 512, 512, 1024]
    cutoff: [1.e-10]
    noise: [1.e-3, 1.e-4, 1.e-5, 1.e-6, 0.]
```

As a table, whose last row repeats:

```yaml
dmrg_gpu:
  sweeps:
    nsweeps: 14
    table: |
      maxdim mindim cutoff noise
      32     8      1e-10  1e-3
      256    8      1e-10  1e-5
      2048   8      1e-10  0
```

The schedule can also be set from the command line (`--nsweeps --maxdim --mindim --cutoff --noise
--krylovdim`) or read from a table file with `--sweeps-table FILE`.

Notes:

- `noise` repeats its last value like the other parameters, so end it with `0`.
- The `primal` and `primal-sym` models need noise in the first sweeps. Their three-site plaquette
  terms otherwise leave two-site DMRG stuck: near the initial product state (`primal`), or with
  frozen block dimensions (`primal-sym`). For `primal-sym` the noise is a block-sparse version of
  White's correction, so it never leaves the sector.
- `energy_tol` behaves like ITensors' `DMRGObserver(energy_tol=...)`. It stops early once the
  energy changes by less than `energy_tol`, but only on a sweep without noise and when `maxdim`
  does not grow in the next sweep.

## Python API

```python
from skqd_z2lgt.dmrg_gpu.backend import xp
from skqd_z2lgt.dmrg_gpu.models import make_lattice
from skqd_z2lgt.dmrg_gpu.runner import solve
from skqd_z2lgt.dmrg_gpu.sweeps import Sweeps

xp.configure('cupy', device=0)            # or 'numpy'
lattice = make_lattice((4, 7))            # or a configuration string
res = solve(lattice, 1.2, [9, 13], model='dual',
            sweeps=Sweeps(nsweeps=10, maxdim=[16, 32, 128], cutoff=1e-12,
                          noise=[1e-5, 1e-6, 0.]))
res['energy'], res['mps_bond_dims'].max(), res['history'][-1]
```

`solve()` returns a dict with these keys:

- `energy`: physical ground-state energy.
- `dmrg_energy`: includes the penalty.
- `penalty_energy`
- `gauss_expectations`, `gauss_targets`: primal model only.
- `mps`: MPS tensors in chain order, mapped by `order`.
- `mpo_bond_dims`, `mps_bond_dims`
- `history`: per-sweep records (including `variance` with `measure_variance=True`).
- `variance`: `<H^2> - <H>^2` of the final state.
- `extrapolation`: see below.
- `state`: the final state (dense MPS list, or `SymmetricMPS` for `primal-sym`), reusable as
  `initial_state`.
- `samples`: only when `num_samples > 0`. Bit strings indexed by qubit; column `q` is qubit `q`
  of the SparsePauliOp.

Sampled primal bit strings satisfy Gauss's law of the sector.

### Error estimates: variance and extrapolation

`variance = <H^2> - <H>^2` vanishes for an exact eigenstate and is always reported. With
`measure_variance=True` it is also recorded after every sweep, and `extrapolation` fits the energy
linearly against the truncation error and against the variance. The fit uses the final noiseless
sweep of each `maxdim` with a positive truncation error. It returns:

- `energy_best`: lowest variational energy;
- `truncation`, `variance`: fit results;
- `energy_extrapolated`
- `error = |energy_best - extrapolated|`

The error is a conservative error bar, not a statistical one. To get several fit points, use a
schedule that ends with noiseless sweeps at increasing `maxdim`, e.g. `maxdim: [..., 128, 128,
192, 192, 256, 256]`.

### Checkpoint / restart

`solve(..., checkpoint='run.npz')` writes the MPS, sweep history and problem metadata after every
sweep (atomically). `resume=True` continues with the remaining sweeps of the schedule. It refuses a
checkpoint of a different problem (model, lattice, coupling, charges, ordering, penalty).

### Coupling scans with warm starts

```python
from skqd_z2lgt.dmrg_gpu.scan import scan
results = scan(lattice, [0.8, 1.0, 1.2, 1.4], [9, 13], model='dual', cold_every=2,
               bidirectional=True)
```

Each point starts from the converged state of the previous coupling. Warm starts can stay trapped
in the previous minimum, so the scan guards against it:

- the inherited state is randomly perturbed (`perturbation`, default 1e-2);
- the warm schedule (`warm_sweeps`) starts with noise;
- cold starts are run at the first and last point and every `cold_every`-th point, and the scan
  continues from the lower state;
- `bidirectional=True` adds a backward warm pass.

Per point, `scan()` returns:

- `energy`: the minimum over all runs (DMRG is variational);
- `source`: which run gave it;
- `energies`: the energies of all runs;
- `flags`: a warning whenever a warm start ended above a cold start or the forward and backward
  passes disagree by more than `tolerance`.

Treat flagged points with suspicion and rerun them cold with larger `maxdim`.

### Excited states (optional)

`solve(..., nstates=2)` also returns `energies[1]`, so the gap `E1 - E0` within the sector is
available. Each further state is found by DMRG with the lower states projected out, as in
ITensors' `dmrg(H, [psi0], psi; weight)` (`excited_weight`, default 20). This is the DMRG
counterpart of `scripts/gaps.py`, which does the same with exact state vectors and is therefore
limited to about 32 plaquettes. Ground-state runs are unaffected (`nstates=1` by default).

For the primal model, `penalty` must also exceed the gaps of interest.

## Output (HDF5)

```
configuration/   lattice, plaquette_energy, charged_vertices, dmrg (JSON incl. sweeps), backend
dual/ primal/ primal-sym/
                 energy, dmrg_energy, penalty_energy, variance, extrapolation (JSON),
                 gauss_expectations, gauss_targets, order, mpo_bond_dims, mps_bond_dims,
                 link_state,
                 history/{sweep, maxdim, cutoff, noise, energy, max_bond, max_trunc_err, seconds,
                          variance},
                 samples (packed bits, attrs num_bits; if requested), mps/<site> (if --save-mps)
```

Standard output carries one JSON line per sweep and a summary per model. The summary includes
the variance and, if available, the extrapolated energy with its error estimate. When the dual
model was run, the output ends with `primal_minus_dual` / `primal-sym_minus_dual`.

## Validation

```bash
python -m pytest tests/test_dmrg_gpu.py
```

These tests run on CPU or GPU. They check:

- the MPO against dense matrices;
- dual and primal DMRG against exact diagonalization (vacuum and charged; K = 0.5, 1, 2);
- `primal-sym` against exact diagonalization, with `<G_v> = q_v` to machine precision;
- the variance, and that the extrapolation error bar covers the actual error;
- checkpoint/resume for all three models, including refusal of a mismatched checkpoint;
- warm-start scans in both directions;
- the snake ordering, and that every ordering gives the exact energy;
- the `eigh` truncation path;
- Gauss's law of sampled bit strings;
- the sweep-schedule parsing.

```bash
python scripts/dmrg_charged_sector_check.py --exact 4x4 4x5 4x6 --scan 4x7 -o charged.h5
```

This script tests the long-string charged sector (charges at the two ends of the middle row, e.g.
vertices 9 and 13 on 4x7):

- E0 and E1 of both models against matrix-free exact diagonalization on small lattices;
- a lambda scan on 4x7 with convergence in the bond dimension, the gap, and the vacuum energy.

Before relying on a production result, check:

- the change of the energy with `maxdim`;
- `max_trunc_err` of the final sweeps;
- `variance` (should be small) and the extrapolation `error_estimate`;
- `primal_minus_dual` / `primal-sym_minus_dual` (should go to 0);
- for the `primal` model, `penalty_energy` and `max_gauss_violation` (should be near 0; exactly 0
  for `primal-sym`).

## Performance notes

Measured on CPU (laptop), full 156q lattice (64 plaquettes, 108 links, 45 vertices):

- **Dual model, K = 1:** converges at bond dimension about 90 to 130, in about 1 to 2.5 s per sweep:

  | sector | E0 |
  |---|---|
  | vacuum | -118.7595278742 |
  | charges [0, 5] | -117.0416260620 |
  | charges [0, 4] | -112.2193963776 |

- **Primal model:** needs much larger bond dimension, because the links carry redundant gauge
  degrees of freedom. With the snake ordering at bond dimension 128 (penalty version), charges
  [0, 4], it is 2.3e-6 above the dual model. CPU time per sweep (penalty version, earlier RCM
  ordering):

  | bond dimension | 128 | 256 | 384 | 512 |
  |---|---|---|---|---|
  | seconds per sweep | 20 | 40 | 68 | 83 |

- **CPU or GPU:** at bond dimensions of a few hundred or less, the CPU is sufficient and a GPU
  gives little or no speedup, because kernel-launch and synchronization overhead dominate. Use the
  GPU for the primal model at bond dimensions of roughly 500 to 1000 and above, for larger K,
  bigger lattices, or many-point scans. With CuPy, `svd: eigh` is recommended, and cuTENSOR is used
  for contractions when installed.
- **Host synchronization:** the Lanczos solver keeps its Krylov basis and tridiagonal coefficients
  on the device. It synchronizes with the host once per restart instead of on every dot product.
- **`primal-sym` cost:** at most 2^(open vertices) labels per bond (at most 2^6 = 64 on 156q with
  the snake). On 156q (CPU) it is about 15 to 60 times cheaper per sweep than the penalty version:
  - bond dimension 128: 2.5 s against 44 s per sweep;
  - bond dimension 256: 30 s in total for an error of 8e-8, against 5+ minutes for 2e-6 with the
    penalty version at bond dimension 128.

  It is the recommended primal cross-check. On GPU, expect it to pay off only at large total bond
  dimension.
- **Parallel runs:** run independent couplings as separate processes with
  `OPENBLAS_NUM_THREADS=1`. A single job mostly uses one core.
- **Ordering:** the default row snake beats reverse Cuthill-McKee by a factor of about 500 in
  accuracy for the primal model at fixed bond dimension. See
  [docs/dmrg_method.md](../../docs/dmrg_method.md), Section 6.
- **MPO size:** the MPO is built exactly by a finite-state automaton and then compressed losslessly
  by SVD. With the snake ordering its bond dimension is 6 for the dual model and 13 for the primal
  model on the 156q lattice.

## Limitations

- The primal model does not support lattices with inactive plaquettes. Their conserved fluxes would
  need extra nonlocal penalty terms. The dual model handles them through `b`.
- Only real Hamiltonians with X and Z terms are supported (no Y), which covers this model.
- `primal-sym`: at most 62 vertices (bit masks in int64), no excited states (`nstates=1`).
- Single-GPU only.

## Files

| file | content |
|---|---|
| `backend.py` | NumPy/CuPy switch `xp` (same pattern as the dissipative iPEPS project) |
| `mpo.py` | Pauli-sum to MPO (automaton + SVD compression), site ordering (snake / RCM) |
| `models.py` | dual and primal problems from heavyhex-qft lattices (with qubit positions and Gauss-law supports) |
| `symmetric.py` | block-sparse exact-Gauss-law MPS and two-site DMRG (`primal-sym`) |
| `analysis.py` | energy extrapolation in truncation error / variance |
| `checkpoint.py` | checkpoint save/load, compatibility check, state perturbation |
| `scan.py` | coupling scans with guarded warm starts |
| `sweeps.py` | ITensors-style `Sweeps` |
| `dmrg.py` | MPS, environments, low-sync Lanczos, truncation, noise, two-site DMRG, excited-state projection, variance, sampling |
| `runner.py` | `solve()`: build, run, measure |
| `scripts/run_dmrg_gpu.py`, `yaml/dmrg_gpu_conf.yaml` | command line and example configuration |
| `scripts/dmrg_charged_sector_check.py`, `tests/test_dmrg_gpu.py` | validation |
