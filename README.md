# Sample-based Krylov quantum diagonalization of two-dimensional Z2 lattice gauge theory

This package provides an implementation of sample-based Krylov quantum diagonalization (SKQD)
applied to pure two-dimensional Z2 lattice gauge theory (LGT) problems.

## Quick Start

### 1. Clone the repo, create a venv, install the package and dependencies

```bash
git clone git@github.com:yiiyama/skqd_z2lgt
cd skqd_z2lgt
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Set up a Prefect UI and register qcsc-prefect blocks

Find instructions for setting up the UI elsewhere. Free tier of Prefect Cloud will be perfectly fine. Once the UI is ready, connect the workspace by ``prefect login``.

Then register the blocks with
```bash
uv run prefect block register -m qcsc_prefect_blocks.common.blocks
uv run prefect block register -m qcsc_prefect.integrations.qiskit.blocks
```

### 3. Create the necessary blocks

The SKQD workflow will use the following Prefect blocks:

1. CommandBlock for Python
1. (Optional) CommandBlock for Python in CUDA-enabled environment
1. ExecutionProfileBlock for running Python on CPU resource as a single-node job
1. ExecutionProfileBlock for running Python on GPU resource as a single-node job
1. ExecutionProfileBlock for running Python on GPU resource as an MPI job
1. HPCProfileBlock for the classical compute infrastructure
1. QiskitRuntimeConfig for the IBM backend of choice

The blocks can be set up from the browser interface or programmatically as:

```python
import sys
from qcsc_prefect_blocks.common.blocks import (
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)
from qcsc_prefect.integrations.qiskit.blocks import QiskitRuntimeConfig

# 1.
CommandBlock(
    command_name="python",
    executable_key="python",
    description="Python",
).save("python", overwrite=True)

# 2. (Required when e.g. GPU and CPU resources have different CPU architectures)
CommandBlock(
    command_name="python",
    executable_key="python-cuda",
    description="Python in CUDA env",
).save("python-cuda", overwrite=True)

# 3.
ExecutionProfileBlock(
    profile_name="python-cpu-single",
    command_name="python",
    resource_class="cpu",
    launcher="single"
).save("python-cpu-single", overwrite=True)

# 4.
ExecutionProfileBlock(
    profile_name="python-gpu-single",
    command_name="python",
    resource_class="gpu",
    launcher="single"
).save("python-gpu-single", overwrite=True)

# 5.
ExecutionProfileBlock(
    profile_name="python-gpu-mpi",
    command_name="python",
    resource_class="gpu",
    launcher="mpirun"
).save("python-gpu-mpi", overwrite=True)

# This part completely depends on your setup
executable_map = {"python": sys.executable}
executable_map |= {"python-cuda": sys.executable.replace("x86_64", "aarch64")}

# 6.
HPCProfileBlock(
    hpc_target="miyabi",
    queue_cpu="regular-c",
    queue_gpu="regular-g",
    project_cpu="your-project-id",
    project_gpu="your-project-id",
    executable_map=executable_map
).save("hpc-miyabi", overwrite=True)

# 6.alt
HPCProfileBlock(
    hpc_target="local",
    executable_map=executable_map
).save("local", overwrite=True)

# 7.
QiskitRuntimeConfig(
    backend_name="ibm_boston",
    channel="ibm_quantum_platform",
    instance="your-quantum-instance",
    token="your-api-key"   
).save("runtime-boston")
```

If you use Prefect for other projects and thus have many defined blocks, you might want to add some identifier (e.g. "skqd-z2lgt") in front of the saved block names.

### 4. Write a job description yaml

The workflow is steered with an ``skqd_z2lgt.orchestration.parameters.Parameters`` object, which can be set up from a yaml file. Many parameters have default values, so a minimal setup would be something like

**parameters_full156_boston.yaml**
```yaml
---
lgt:
  lattice: full_156q  # Lattice that uses all 156 qubits of Heron r2/r3
circuit:
  sampled_steps: [2, 3, 4, 5]  # Sample from 2-, 3-, 4-, and 5-Trotter step circuits
runtime:
  config_block: runtime-boston
  shots_exp: 10000
  shots_ref: 20000
python_cuda_cmd: python-cuda
hpc_profile: hpc-miyabi
pkgpath: /full/path/to/output/directory
```

``Parameters.lgt.lattice`` accepts either one of

- "full_156q"
- String depicting the lattice (see [TriangularZ2Lattice docstring](https://github.com/UTokyo-ICEPP/heavyhex-qft/blob/main/heavyhex_qft/triangular_z2.py) for details)
- A 2-tuple of ints specifying the rectangular arrangement of triangular plaquettes
- A path to a JSON file created with the ``to_json()`` method of the lattice object

If the Prefect blocks above were saved under different names, they must be specified with top-level parameters ``python_cmd``, ``python_cuda_cmd``, ``python_cpu_single_exec_profile``, ``python_gpu_single_exec_profile``, and ``python_gpu_mpi_exec_profile``.

The directory ``pkgpath`` points to will contain everything that the workflow produces.

### 5. Run the workflow

```bash
skqd-z2lgt parameters_full156_boston.yaml
```

### 6. Study the output

Important outputs in ``pkgpath`` are

- ``skqd_cr.h5``: Contains the final SKQD energy estimate (history through iterative subspace determination)
- ``crbm/*.h5``: Contains the CRBM weights and train / test metrics

To first order, we should make sure that CRBMs were properly trained (metric ``test_success_rate`` plateaus at a nontrivial value) and also the energy estimate declines.

## DMRG reference energies (`skqd_z2lgt.dmrg_gpu`)

Classical ground-state energies in the vacuum and charge sectors, from DMRG on CPU or NVIDIA GPU,
for both the dual Ising model (as in SKQD) and the primal link model with a Gauss-law penalty.
See [skqd_z2lgt/dmrg_gpu/README.md](skqd_z2lgt/dmrg_gpu/README.md).
