# DMRG reference calculations for the 2D Z2 LGT: method and implementation

This note summarizes how the classical DMRG reference energies for the SKQD project are computed:
the physics setup, the two formulations, the algorithm, how the 2D lattice is mapped onto the MPS
chain (snake geometry), the implementation, validation and performance. For usage (commands,
configuration, outputs), see [skqd_z2lgt/dmrg_gpu/README.md](../skqd_z2lgt/dmrg_gpu/README.md).

## 1. Purpose

SKQD estimates ground-state energies of the pure Z2 lattice gauge theory on triangular lattices
(up to the 156-qubit heavy-hex embedding, with 64 plaquettes and 108 links), in the vacuum and in
static-charge sectors. Two kinds of reference are needed:

- **Exact values at sizes beyond full state vectors.** The JAX solvers in `scripts/ground_state_*.py`
  and `scripts/gaps.py` are limited to about 32 plaquettes. The previous ITensors.jl DMRG path
  (`skqd_z2lgt/ising_dmrg.py`) required Julia and a sysimage.
- **An independent cross-check of the charge-sector construction**, i.e. that the dual Hamiltonian
  SKQD diagonalizes is indeed the gauge theory in the intended sector.

The DMRG provides both. It runs on CPU (NumPy) or an NVIDIA GPU (CuPy), using the same backend
pattern as the dissipative iPEPS project.

## 2. Model and charge sectors

The Hamiltonian, with one qubit per link `l` of the triangular lattice, is

    H = - sum_l Z_l - lambda sum_p X_l1 X_l2 X_l3        (lambda = lgt.plaquette_energy = K)

- **Gauss law.** The Gauss-law operators `G_v = prod_{l in v} Z_l` commute with `H`.
- **Charge sectors.** A sector is a set of eigenvalues `q_v = +-1`, with `q_v = -1` on the charged
  vertices. In the computational basis, the charge at `v` is the parity of the link bits around `v`.
- **Base link state.** The SKQD workflow fixes the sector through a base link state `b`
  (`minimum_weight_link_state`, a pymatching decoding of the charge configuration). `b` is a
  minimum-length string of excited links joining the charges.

All DMRG calculations take the sector from the same `b`, so they target exactly the sector SKQD
uses.

## 3. Three formulations

### 3.1 Dual Ising model (what SKQD diagonalizes)

`lattice.plaquette_dual(b).make_hamiltonian(lambda)` has one qubit per active plaquette `p`:

    H_dual = sum_{l interior} s_l Z_p Z_p'  +  sum_{l boundary} s_l Z_p  -  lambda sum_p X_p,
    s_l = -1 + 2 b_l

- **Mapping.** A plaquette spin flip corresponds to the plaquette operator `XXX` on its three links,
  and `Z_p Z_p'` is the electric energy `Z_l` of the shared link.
- **Charges.** The sector enters only through the sign flips of the couplings along the string `b`.
- **Exactness.** On an open planar lattice, `V - L + P = 1`, so the plaquette configurations
  enumerate the gauge-invariant states of the sector exactly. `H_dual` therefore has the same
  spectrum as `H` restricted to the sector.

### 3.2 Primal link model with a Gauss-law penalty

    H_pen = H + Lambda sum_v (1 - q_v G_v)/2  =  H + (Lambda/4) sum_v (G_v - q_v)^2

The second form follows from `G_v^2 = 1`. Each violated vertex costs `Lambda`, the `penalty`
parameter, default 4.

- **Why the penalty is not variationally biased.** `H` commutes with every `G_v`, so its
  eigenstates in the target sector are eigenstates of `H_pen` with the same energy. All other
  sectors are shifted up by at least `Lambda`. As long as `Lambda` exceeds the inter-sector energy
  differences of interest, the ground state of `H_pen` is the ground state of `H` in the sector.
- **Starting point.** DMRG starts from the basis state `|b>`, which already lies in the sector. The
  penalty mainly removes leakage caused by truncation and by the noise term (Section 4).
- **Reported quantities.** The physical energy `<H>` and the Gauss-law expectations `<G_v>`.

### 3.3 Primal link model with exact Gauss-law symmetry (`primal-sym`)

All `G_v` commute with each other and with `H`, so the symmetry group is abelian, `Z2^V`. A
computational basis state `s` of link `l` carries parity `s` at the two end vertices of `l`. Its
charge is `s * mask_l`, with `mask_l` the bitmask of those two vertices.

**Bond labels.** Each MPS bond index carries a label: the vertex-parity bitmask of all links to the
left of the bond. Every tensor is block-sparse,

    A[i, s, x, y] != 0  only if  label(right) = label(i) XOR s * mask_site.

**Which labels exist.** The allowed labels of a bond are those reachable from the left boundary
(label 0) that can still be completed to the target charges at the right boundary. At a given cut
each vertex falls into one of three cases:

- all its links lie to the left: its parity must already equal `q_v`, so the Gauss law is enforced
  exactly;
- all its links lie to the right: its parity is 0;
- its links are split across the cut ("open" vertex): its parity is free.

Only open vertices carry information, so a bond has at most `2^(open vertices)` labels. On the 156q
lattice with the snake ordering there are at most 6 open vertices (5.2 on average), i.e. at most 64
labels per bond.

**Storage.** Blocks of one bond are padded to a common size and stored as dense arrays indexed by
label: `A[i, s, x, y]`, environments `L[a, i, X, x]`. The bra label of an environment block is
fixed by the operator charge `delta_a` of the MPO automaton state `a`, i.e. the parity flipped by
the X operators already placed. This is why `primal-sym` uses the uncompressed automaton MPO: SVD
compression would mix automaton states of different charge.

**Algorithm.**

- The two-site matrix-vector product loops over MPO channels `(a, c)`. Each channel is a batched
  GEMM over all labels, followed by a scatter to the bra labels.
- Truncation is done by an SVD (or `eigh`) per middle-bond label, with one global cutoff across all
  blocks.
- No penalty is needed, and the state cannot leak out of the sector at any step.

**Noise.** Two-site DMRG with this symmetry needs noise, for a structural reason. A link whose
state is fixed by the labels on both sides (a vertex closes there) ties together the block
dimensions of the neighbouring bonds. The two-site update then cannot increase them, and the bond
dimension freezes.

- Observed: on a 19-link lattice the bond dimension stayed at 8, against 21 for the exact state,
  with an energy error of about 1e-4.
- Random noise inside the existing blocks does not help, because it adds no new column space.
- White's correction does help, applied per block. For each MPO bond index `b`, the half-system
  operator is applied to theta and added to the reduced density matrix of the middle label shifted
  by `delta_b`. This enlarges the column space through the MPO index. With it the energy is exact
  to 1e-9 to 1e-11.

**Scope.** For energies, `primal-sym` spans exactly the same `2^P`-dimensional space as the dual
model, so it does not beat the dual DMRG. Its value is the link-resolved state: electric flux on
individual links, and link-basis samples comparable with hardware bit strings. It also provides an
independent primal cross-check without penalty bias.

### 3.4 Why several formulations

The formulations share nothing beyond the lattice and `b`, so agreement of their energies
tests:

- the charge-sector construction (`b`, the sign rule, the qubit index conventions of
  heavyhex-qft);
- the DMRG itself.

On 156 qubits, the dual model is the cheap and accurate reference: 64 sites, low entanglement. The
primal model is the independent check: 108 sites, with redundant gauge degrees of freedom that make
it costlier.

## 4. Algorithm

### 4.1 Two-site DMRG

Two-site DMRG with open boundary conditions is standard, so only the choices made here are listed.

- **MPS.** Real tensors `A[l, s, r]`; the Hamiltonians are real (only X and Z), so this halves the
  memory and cost relative to complex arithmetic.
- **MPO.** `W[a, b, s, t]` with left and right environments `L[bra, w, ket]`, `R[bra, w, ket]`.
  The effective two-site matrix-vector product is a sequence of four `tensordot` calls, i.e.
  dense GEMMs (cuBLAS / cuTENSOR on GPU).
- **Local eigensolver.** Restarted Lanczos with full reorthogonalization. Krylov dimension
  `krylovdim` (default 20) and `maxiter` restarts (default 2). It is written to avoid
  host-device synchronization:
  - the Krylov basis is a device matrix `V`;
  - reorthogonalization is block classical Gram-Schmidt, applied twice, i.e. two GEMVs
    `w -= V^T (V w)` per step;
  - alpha and beta stay in device arrays.

  The host reads the tridiagonal coefficients once per restart, instead of `k + 2` scalar transfers
  per Krylov step in a naive implementation. A breakdown (invariant subspace) is detected
  afterwards, and the Krylov space is truncated there.
- **Truncation.**
  - At most `maxdim` states, discarding at most `cutoff` of the weight, keeping at least `mindim`.
    This matches the semantics of ITensors.
  - `svd` uses gesvd; `eigh` diagonalizes the reduced density matrix `theta theta^T`, which is
    much faster on GPU at large bond dimension.
- **Density-matrix perturbation (noise).** White's correction
  `rho -> rho + noise * sum_b (H_b theta)(H_b theta)^T / tr`, with `H_b` the half-system MPO
  components on the kept side, normalized to trace `noise`.
- **Sweep schedule.** Following ITensors' `Sweeps`: per-sweep `maxdim`, `mindim`, `cutoff`,
  `noise`, `krylovdim`, `maxiter`, with the last value repeating. Early stopping (`energy_tol`)
  is only applied on noiseless sweeps whose `maxdim` no longer grows.

### 4.2 Why the noise term is required for the primal model

The plaquette term `XXX` acts on three links. When DMRG starts from a product state, the
three-site term projected onto the two-site window has a vanishing off-diagonal part, because
`<0|X|0> = 0` on the third link. The plain two-site update then does not move.

- **Without any fix**, the energy stayed exactly at `<b|H|b>`.
- **With only a random admixture in the initial state**, DMRG converged to a wrong state with bond
  dimension 4, about 1.0 above the exact energy on a 19-link lattice. This local minimum is
  stabilized by the many local Z2 symmetries.
- **With the noise term**, the energy is exact to 1e-10.

The default schedule therefore starts with `noise = 1e-3` and decays it to 0. The dual model has
only one- and two-site terms and converges without noise, but a small noise does no harm.

### 4.3 Error estimates: variance and extrapolation

**Variance.** The energy variance `<H^2> - <H>^2` vanishes for an exact eigenstate. It is
evaluated with a double-layer environment carrying two MPO indices, at a cost per site of about
`chi^3 w^2 d^2`. It is always reported for the final state, and with `measure_variance=True` after
every sweep. For converged states on small lattices it is about 1e-11 to 1e-9.

**Extrapolation.** `analysis.extrapolate` fits `E` linearly against the truncation error (largest
discarded weight of the sweep) and against the variance. It uses the final noiseless sweep of each
`maxdim`: noisy sweeps are excluded because the noise biases both quantities. It reports:

- the best variational energy;
- the extrapolated energy;
- `error = |E_best - E_extrapolated|`, a conservative error bar rather than a statistical one.

Test on 4x7 (charges [9, 13]) with a deliberately small bond dimension (at most 16), compared with
a converged reference:

| lambda | actual error of `E_best` | reported `error` |
|---|---|---|
| 1.2 | 6.5e-5 | 9.7e-5 |
| 2.0 | 2.0e-4 | 3.2e-4 |

The error bar covered the actual error in both cases. The extrapolated values themselves were not
more accurate than `E_best`, because the fit points from strongly truncated runs are still in the
curved regime. Use the extrapolation as an error estimate unless several points with small
truncation errors are available.

### 4.4 Checkpoint/restart and warm starts

**Checkpoints.** A checkpoint (`checkpoint.py`) is one `.npz` file, written atomically after every
sweep. It holds the MPS (dense, or block-sparse with labels and dimensions), the sweep history and
the problem metadata: model, lattice, coupling, charges, ordering, order, penalty and schedule.
`resume=True` runs the remaining sweeps of the schedule. It refuses a checkpoint whose metadata
differs, including the coupling.

**Warm starts.** `scan.scan` reuses the converged state of the neighbouring coupling. A warm start
can stay in the basin of the previous solution, for example across level crossings or where the
ground state changes character. Safeguards:

1. The inherited state is perturbed randomly, by 1e-2 relative per site. For `primal-sym` the
   perturbation is restricted to the allowed blocks.
2. The warm schedule starts with noise, i.e. White's correction, which is the mechanism that lets
   DMRG leave a basin.
3. Cold starts run at the first and last coupling and every `cold_every`-th point. If a cold start
   is lower than the warm one by more than `tolerance`, the point is flagged and the scan continues
   from the cold state.
4. With `bidirectional=True`, a backward warm pass also runs. The per-point minimum is kept and the
   hysteresis is flagged.

All run energies and flags are returned. A disagreement is never resolved silently, since a flag is
the signal of a possible wrong minimum.

### 4.5 Excited states (optional)

`nstates > 1` finds further states with the lower ones projected out, i.e. it minimizes
`H + w sum_k |psi_k><psi_k|`, as in ITensors' `dmrg(H, [psi0], psi; weight)`.

- **Implementation.** The projector enters the local matrix-vector product through overlap
  environments `<mps|psi_k>`, updated during the sweep.
- **Use.** This is the DMRG counterpart of `scripts/gaps.py`, which computes the gap `E1 - E0` with
  state vectors. It is only needed for gaps beyond about 32 plaquettes. SKQD itself does not use
  it.

### 4.6 Measurements

- The physical energy `<H>`, from a separate MPO without penalty or projector.
- `<G_v>`, from bond-dimension-1 Z-string MPOs.
- Overlaps between states.
- Perfect sampling of bit strings from the right-canonical MPS, vectorized over samples. Samples
  from the primal model satisfy the Gauss law of the sector, which is tested.

## 5. MPO construction

Each Hamiltonian, including the penalty terms, is a sum of Pauli strings with X and Z only. The MPO
is built exactly and then compressed.

1. **Finite-state automaton.** The automaton states on the bond between sites `j-1` and `j` are:
   - `start`: no operator placed yet;
   - `final`: the term is complete;
   - for each open term, the tuple of operators already placed.

   Terms with a common prefix share states. The coefficient is attached to the last operator of
   each term. Unreachable states are removed.
2. **Lossless SVD compression.** Two left-right sweeps, dropping singular values below `1e-13`
   relative to the largest. This removes exact linear dependencies, e.g. common suffixes.

On the 156q lattice with the snake ordering, the MPO bond dimension is at most 6 for the dual model
and at most 13 for the primal model including the Gauss-law terms. The Gauss-law terms are six-body
Z strings in the bulk. Correctness is tested against dense matrices for random Pauli sums.

## 6. Mapping the 2D lattice onto the chain: snake geometry

### 6.1 Why the ordering matters

An MPS is one-dimensional. Every bond of the chain cuts the 2D lattice into two regions, and the
bond dimension needed scales with the entanglement across that cut. In the area-law regime,
entanglement grows with the length of the cut's boundary. Two quantities follow the ordering:

- **Cut size:** the number of Hamiltonian terms crossing a cut. It bounds the MPO bond dimension
  and is a proxy for the boundary length.
- **MPS bond dimension** needed for a given accuracy.

### 6.2 Geometry of the lattice

Each qubit has a 2D position, taken from the lattice data of heavyhex-qft:

- **Dual model:** the centre of the plaquette. The triangles of one row alternate between pointing
  up and pointing down, at consecutive x. On the 156q lattice there are 8 rows (y = 0.5 ... 7.5) of
  8 triangles (x = 13 ... 20).
- **Primal model:** the centre of the link. Horizontal links lie at integer y, diagonal links at
  half-integer y, so they form separate rows.

### 6.3 Snake (boustrophedon) ordering

The sites are grouped into rows of equal y, with y rounded to half units. Rows are taken in
increasing y, and successive rows are traversed in alternating x direction. For the dual model
(plaquettes, `^` = up-triangle, `v` = down-triangle):

```
 y=7.5   56 57 58 59 60 61 62 63  ->
 y=6.5  <- 55 54 53 52 51 50 49 48
  ...
 y=1.5  <- 15 14 13 12 11 10  9  8
 y=0.5    0  1  2  3  4  5  6  7  ->
          ^  v  ^  v  ^  v  ^  v     (alternating triangles along x)
```

The numbers are positions along the MPS chain.

- **Turns.** The chain turns at the end of each row, so the last site of one row and the first site
  of the next are spatial neighbours. A "typewriter" ordering would instead jump across the whole
  lattice at each row end.
- **Cuts.** A cut after chain position `k` separates the rows below plus the first part of the
  current row from the rest. Its boundary is about one row long, independent of where in the row
  it falls.
- **Direction choice.** The other direction ('columns': columns of equal x, traversed alternately
  in y) is also possible. The default `snake` builds both and keeps the one with the smaller
  maximum cut size, ties broken by the mean.

### 6.4 Why the row direction wins on the triangular lattice

- **Rows** of the triangular lattice are straight strips of edge-sharing triangles.
- **Columns** at fixed x mix up- and down-triangles, which do not share edges with each other
  across the column. A column cut is therefore a zigzag line crossing many more couplings.

The difference between the two snake directions is thus a property of the lattice, not only of its
aspect ratio. The automatic choice handles it.

### 6.5 Reverse Cuthill-McKee (alternative)

Reverse Cuthill-McKee (RCM) is a geometry-free sparse-matrix ordering, and it was the default in the
first version.

1. Start a breadth-first search at a peripheral, low-degree node of the interaction graph. Two
   qubits are connected if they share a Hamiltonian term.
2. Number the nodes in BFS order, with lower-degree neighbours first.
3. Reverse the order.

The result minimizes the bandwidth `max |i - j|` over interacting pairs. Its advantage is that it
needs no coordinates, so it works for any lattice. On a near-square lattice, however, BFS level
sets are diagonal wavefronts, which are longer than a row. It remains available as
`ordering: rcm`, and the code falls back to it if positions are missing.

### 6.6 Measured comparison

156q lattice, lambda = 1, charges [0, 4]. Converged reference energy: E = -112.2193963776 (dual, bond
dimension about 130, truncation 1e-14). Each ordering was run at a fixed bond dimension.

Dual model (64 sites), bond dimension 64:

| ordering | max MPO bond | max / mean cut | E - E_ref | time |
|---|---|---|---|---|
| natural (heavyhex ids) | 7 | 6 / 4.4 | 2.7e-9 | 5 s |
| RCM | 8 | 9 / 5.7 | 0.9e-9 | 6 s |
| snake, columns | 10 | 9 / 7.6 | 2.8e-5 | 16 s |
| **snake, rows (default)** | **6** | **5 / 4.0** | **2e-10** | **5 s** |

Primal model (108 sites), bond dimension 128:

| ordering | max MPO bond | max / mean cut | E - E_ref | time |
|---|---|---|---|---|
| natural | 13 | 11 / 9.4 | 1.2e-5 | 144 s |
| RCM | 17 | 15 / 11.0 | 1.2e-3 | 199 s |
| snake, columns | 19 | 17 / 16.0 | 9.4e-2 | 237 s |
| **snake, rows (default)** | **13** | **11 / 9.4** | **2.3e-6** | **137 s** |

The row snake is the most accurate ordering in both models.

- **Primal model.** It is about 500 times more accurate than RCM at the same bond dimension.
- **Natural ordering.** The heavyhex-qft ids happen to be close to a row-by-row typewriter order.
  They therefore have the same cut statistics as the row snake, but lose about a factor of 5 in
  accuracy, presumably because of the jumps at the row ends.

## 7. Implementation

Package `skqd_z2lgt/dmrg_gpu/`:

| file | role |
|---|---|
| `backend.py` | `xp` proxy to NumPy or CuPy. Device selection, precision, cuTENSOR accelerator. Same pattern as `lp_ipeps/backend.py` in the dissipative project. |
| `mpo.py` | Pauli terms from `SparsePauliOp`, automaton MPO, SVD compression, orderings (`snake`, `rcm`, `natural`), cut sizes |
| `models.py` | dual and primal problems from heavyhex-qft lattices. Base link state via `minimum_weight_link_state`, Gauss-law supports and targets, qubit positions. |
| `sweeps.py` | ITensors-style `Sweeps` (lists or table) |
| `dmrg.py` | MPS initialization, environments, low-sync Lanczos, truncation (`svd` / `eigh`), noise, two-site DMRG, excited-state projection, expectation values, variance, sampling |
| `symmetric.py` | `primal-sym`: Gauss-law labels (`GaussLawChain`), block-sparse `SymmetricMPS`, symmetric environments, block-wise White correction, two-site DMRG |
| `analysis.py` | extrapolation in truncation error / variance |
| `checkpoint.py` | checkpoint save/load, compatibility check, state perturbation |
| `scan.py` | coupling scans with guarded warm starts |
| `runner.py` | `solve()`: build problem, order, MPO, initial state (or `initial_state` / resume), run, measure |

Scripts and configuration:

- `scripts/run_dmrg_gpu.py` with `yaml/dmrg_gpu_conf.yaml`: production runs, HDF5 output.
- `scripts/dmrg_charged_sector_check.py`: charged-sector validation and lambda scans.
- `tests/test_dmrg_gpu.py`: unit and exact-diagonalization tests.

Conventions worth knowing:

- **Bit order.** heavyhex-qft link-state arrays are bit-reversed relative to the qubit index: array
  position `i` holds qubit `nl - 1 - i`. `models.link_bits_by_qubit` converts them. The Gauss-law
  supports are built from `lattice._lid_to_bit`, the same map used by `make_hamiltonian`. A test
  checks that every `G_v` commutes with every plaquette term and that the -1 targets fall on the
  requested vertices.
- **Sample columns.** Samples are returned indexed by the qubit of the corresponding
  `SparsePauliOp`, i.e. in the same convention as SKQD's dual bit strings.
- **GPU work.** All tensor work runs on the device. Only the following run on the host: MPO
  construction (small), the tridiagonal Lanczos problem, and truncation bookkeeping on singular
  values.

Limitations:

- The primal model does not support inactive plaquettes. Their fluxes would need nonlocal
  penalties; the 156q lattice has none.
- `primal-sym` supports at most 62 vertices (bit masks in int64) and only the ground state.
- Single GPU only.

## 8. Validation

### 8.1 Unit tests

`tests/test_dmrg_gpu.py`, 22 tests:

- the MPO against dense matrices, with and without compression, including the penalty terms;
- dual and primal DMRG against exact diagonalization, vacuum and charged, lambda = 0.5, 1, 2.
  Agreement is better than 1e-7, with zero penalty energy and `<G_v>` equal to the targets;
- the penalized primal spectrum against the dual one by brute force;
- the `eigh` truncation and Gauss's law of sampled bit strings;
- the sweep schedule;
- the snake geometry, and that all orderings give the exact energy;
- `primal-sym` against exact diagonalization, with `|<G_v> - q_v| < 1e-12` and a small variance;
- the variance, and that the extrapolation error bar covers the actual error;
- checkpoint/resume for `dual`, `primal` and `primal-sym`, including refusal of a checkpoint at a
  different coupling;
- warm-start scans, forward and backward, against exact energies with no false flags.

Further `primal-sym` checks against exact diagonalization:

| lattice | charges | lambda = 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 19 links | vacuum | 7e-11 | 1e-10 | 1e-10 |
| 19 links | [0, 4] | 2e-11 | 1e-10 | 3e-12 |
| 4x5 (37 links) | [7, 10] | 8e-10 | 1e-9 | 5e-9 |

In all of these, `<G_v> = q_v` exactly.

### 8.2 Charged-sector check

`scripts/dmrg_charged_sector_check.py` uses the long-string sector: charges at the two ends of the
middle row, e.g. vertices 9 and 13 on 4x7. It compares E0 and E1 from both models, with two states,
against matrix-free exact diagonalization of the dual model.

| lattice | sector | lambda | E0 exact | E1 - E0 | dual dE0, dE1 | primal dE0, dE1 |
|---|---|---|---|---|---|---|
| 4x4 | vacuum | 1.0 | -32.674701 | 4.951 | 6e-11, 7e-11 | 2e-10, 2e-10 |
| 4x4 | [6, 8] | 1.0 | -29.759068 | 1.671 | 8e-11, 8e-11 | 2e-10, 2e-10 |
| 4x4 | [6, 8] | 2.0 | -39.538203 | 2.366 | 1e-10, 1e-10 | 2e-10, 4e-10 |
| 4x5 | vacuum | 0.5 - 2.0 | e.g. -40.346467 at lambda = 1 | 5.66 - 3.31 | <= 2.4e-10 | <= 4.3e-9 |
| 4x5 | [7, 10] | 0.5 | -32.299912 | 1.622 | 1e-10, 1e-10 | 2e-9, 9e-10 |
| 4x5 | [7, 10] | 1.0 | -35.979116 | 1.324 | 2e-10, 2e-10 | 1e-9, 2e-9 |
| 4x5 | [7, 10] | 1.2 | -38.030645 | 1.340 | 2e-10, 2e-10 | 2e-9, 2e-9 |
| 4x5 | [7, 10] | 1.5 | -41.628593 | 1.487 | 2e-10, 2e-10 | 2e-9, 3e-9 |
| 4x5 | [7, 10] | 2.0 | -48.752976 | 1.987 | 3e-10, 2e-10 | 1e-9, 1e-9 |
| 4x6 | vacuum | 0.5 | -44.001433 | 5.653 | 1e-11, 2e-10 | 9e-10, 9e-10 |
| 4x6 | vacuum | 1.0 | -47.022154 | 4.835 | 3e-10, 3e-10 | 9e-10, 3e-9 |
| 4x6 | vacuum | 1.2 | -48.804773 | 4.445 | 4e-10, 4e-10 | 2e-9, 3e-9 |
| 4x6 | vacuum | 2.0 | -59.254961 | 3.074 | 3e-10, 3e-10 | 3e-9, 6e-9 |
| 4x6 | [8, 11] | 0.5 | -38.472389 | 1.582 | 1e-10, 2e-10 | 9e-10, 4e-9 |
| 4x6 | [8, 11] | 1.0 | -42.714440 | 1.223 | 3e-10, 3e-10 | 1e-9, 2e-9 |
| 4x6 | [8, 11] | 1.2 | -45.111422 | 1.225 | 3e-10, 3e-10 | 2e-9, 3e-9 |
| 4x6 | [8, 11] | 1.5 | -49.348327 | 1.358 | 3e-10, 3e-10 | 3e-9, 4e-9 |
| 4x6 | [8, 11] | 2.0 | -57.818218 | 1.849 | 4e-10, 4e-10 | 5e-9, 5e-9 |

The penalty energy of all primal states was at most about 1e-11.

**4x7 lambda scan**, long string [9, 13] (the geometry of the project slides), dual DMRG with two
states:

| lambda | E0 | E1 - E0 | E0 vacuum | bond dimension needed |
|---|---|---|---|---|
| 0.1 | -42.07325019 | 1.97037 | -50.04666944 | 9 |
| 0.3 | -42.65390571 | 1.76786 | -50.42022476 | 17 |
| 0.5 | -43.79366036 | 1.48382 | -51.16839717 | 29 |
| 0.7 | -45.46885144 | 1.22812 | -52.29328795 | 47 |
| 0.9 | -47.65986846 | 1.06306 | -53.79796996 | 61 |
| 1.1 | -50.33524328 | 1.00629 | -55.68631933 | 72 |
| 1.3 | -53.44519335 | 1.04275 | -57.96264083 | 80 |
| 1.5 | -56.93793333 | 1.15170 | -60.63087648 | 86 |
| 1.7 | -60.76789767 | 1.32004 | -63.69294809 | 88 |
| 1.9 | -64.89239450 | 1.53999 | -67.14531773 | 88 |
| 2.1 | -69.26859074 | 1.80457 | -70.97271461 | 85 |

- **Convergence in bond dimension.** Going from 128 to 256 changes neither E0 nor the gap (both
  0.0). The largest bond dimension that actually occurs is 88. Going from 64 to 256 changes E0 by
  at most 4e-9.
- **Primal cross-check.** The penalty primal model agrees with the dual model to 1.9e-8 at
  lambda = 1.1 and 3.3e-8 at lambda = 1.5.
- **Confinement potential.** `V = E0(string) - E0(vacuum)` can be read off directly, e.g. 5.351 at
  lambda = 1.1.

**Finding: the slide gaps are most likely gaps to the lowest mirror-symmetric excitation.**

The gap computed here is lower than the gap curves in the project slides:

| lattice | here: minimum gap | slides: minimum gap |
|---|---|---|
| 4x5 | 1.32 near lambda = 1 | about 1.55 near lambda = 0.8 |
| 4x7 | 1.006 near lambda = 1.1 | about 1.25 near lambda = 0.9 |

The 4x7 geometry here is exactly the one in the slides (charges 9 and 13), so a different charge
placement cannot explain this. Exact diagonalization of 4x5 (charges [7, 10], lambda = 1) shows
the symmetry of the lowest states under the two lattice mirrors:

| state | E - E0 | left-right parity | top-bottom parity |
|---|---|---|---|
| E0 | 0 | + | + |
| E1 | 1.3238 | + | - |
| E2 | 1.6116 | + | + |
| E3 | 1.9899 | - | - |

- **The true first excited state is odd under the top-bottom mirror:** the string bending up minus
  bending down.
- **`scripts/gaps.py` starts in the symmetric sector.** It starts its second eigenvector from the
  uniform superposition of single plaquette flips, which is even under both mirrors. An iterative
  solver started there remains in the fully symmetric sector, so it converges to E2, not E1.
- **E2 matches the slides.** E2 - E0 = 1.61 at lambda = 1 agrees with the 4x5 slide curve.

This has been verified on 4x5 by exact diagonalization only, not by rerunning `gaps.py`. If it is
confirmed, the target-point criterion `|E_PT - E0| > |E1 - E0|` should use the true gap, which is
smaller.

### 8.3 Full 156q lattice

lambda = 1. Dual model, converged in bond dimension (64 to 256; changes below 3e-9):

| sector | E0 | bond dimension needed |
|---|---|---|
| vacuum | -118.7595278742 | about 90 |
| charges [0, 5] | -117.0416260620 | about 90 |
| charges [0, 4] | -112.2193963776 | about 130 |

Primal models, charges [0, 4]. Same schedule for both, reference E0 from the dual model:

| model | bond dimension | E - E0 | variance | max \|<G_v> - q_v\| | total time | time of last sweep |
|---|---|---|---|---|---|---|
| `primal-sym` | 128 | 4.2e-6 | 6.6e-5 | 0 (exact) | 23 s | 2.5 s |
| `primal` (penalty) | 128 | 1.9e-6 | 2.9e-5 | 3e-9 | 323 s | 44 s |
| `primal-sym` | 256 | 8.2e-8 | 1.7e-6 | 0 (exact) | 30 s | 3.2 s |
| `primal` (penalty) | 256 | 7.4e-9 | 1.4e-7 | 2e-11 | 684 s | 195 s (*) |

(*) This run overlapped with other jobs, so its time is inflated. The factor of about 17 at bond
dimension 128 was measured without overlap.

- **At equal cost,** `primal-sym` is far more accurate. In 30 s it reaches 8e-8; the penalty
  version needs more than 5 minutes for 2e-6.
- **At equal bond dimension,** the penalty version is 2 to 10 times more accurate after the same
  short schedule. Both should reach the same optimum within the sector, so this most likely
  reflects how fast each converges, and is not yet understood.
- **Practical use:** run `primal-sym` at a larger bond dimension.

With the earlier RCM ordering, the penalty version at bond dimension 256 was 2.5e-4 above the dual
model.

## 9. Performance and the role of the GPU

- **Dual model on 156q.** It needs bond dimension of order 100. A sweep takes 1 to 2.5 s on a
  laptop CPU, so a GPU brings no benefit: kernel-launch and synchronization overheads dominate the
  small matrices. Use `--backend numpy`.
- **Primal model.** It needs larger bond dimension. CPU time per sweep, measured with the earlier
  RCM ordering:

  | bond dimension | 128 | 256 | 384 | 512 |
  |---|---|---|---|---|
  | seconds per sweep | 20 | 40 | 68 | 83 |

  The cost grows slower than `chi^3`, i.e. it is not yet dominated by large dense linear algebra.
- **When to use the GPU.** It is expected to pay off from a bond dimension of about 500 to 1000
  upwards, which is relevant for the primal model at larger lambda, larger lattices or long scans.
  Use `svd: eigh` there. No custom CUDA kernels are used or needed: nearly all the work is
  GEMM-like contractions and dense eigensolvers, which CuPy already hands to cuBLAS, cuTENSOR and
  cuSOLVER.
- **Host synchronization.** The Lanczos solver synchronizes once per restart (Section 4.1), which
  matters on GPU, where every scalar transfer stalls the stream.
- **`primal-sym`.** Up to 64 labels per bond on 156q. Even so, on CPU it is about 15 to 60 times
  cheaper per sweep than the penalty version, because each label block is much smaller than the
  full bond dimension (Section 8.3). On GPU the many small blocks mean per-launch overhead, so it
  pays off only at large total bond dimension.
- **Parallel runs on CPU.** A single DMRG job mostly uses one core: small matrices, and Python
  loops for `primal-sym`. Independent couplings or lattices are best run as separate processes with
  `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS=1`), to avoid oversubscribing threads. The 4x7
  scan and the 4x6 checks above ran as 17 parallel processes.
- **Not yet measured.** No GPU timings were taken; the development machine has no NVIDIA GPU.

## 10. Recommendations

1. **Reference energies.** Use the dual model with the default snake ordering. Check convergence
   in `maxdim` and `max_trunc_err` of the final sweeps.
2. **Independent check.** Run `primal-sym` (exact Gauss law) or the penalty `primal` model at a
   few points and require `primal*_minus_dual -> 0` with growing `maxdim`. For the penalty
   version, also require `penalty_energy ~ 0` and `max |<G_v> - q_v| ~ 0`.
3. **Error bars.** Report `variance` and the extrapolation `error` alongside every energy. Use a
   schedule with several noiseless sweeps at increasing `maxdim` at the end.
4. **Scans.** Use `scan()` with cold anchors (`cold_every`) and, for production,
   `bidirectional=True`. Rerun flagged points cold with larger `maxdim`.
5. **Gaps.** `nstates=2` finds the true E1 regardless of symmetry. Before quoting gaps from
   `scripts/gaps.py`, check the mirror-symmetry issue of Section 8.2. Its symmetric start vector
   appears to give the gap to the lowest mirror-symmetric excitation instead.
6. **Near the confinement transition** (lambda about 2.1), the entanglement grows. Expect larger
   bond dimension, and this is where the GPU becomes useful.
