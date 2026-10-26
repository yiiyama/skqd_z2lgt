"""Workflow parameters."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LGTParameters(BaseModel):
    """Parameters to specify the Z2 LGT."""

    lattice: str | tuple[int, int] = Field(
        default='full_156q',
        description='Two-dimensional lattice configuration.',
        title='Lattice Configuration'
    )
    plaquette_energy: float = Field(
        default=1.,
        description='Plaquette energy in the Hamiltonian (transverse field strength in the dual'
                    ' Ising model).',
        title='Plaquette Energy',
        ge=0.
    )
    charged_vertices: list[int] = Field(
        default_factory=list,
        description='Vertex ids where static electric charges are to be located.',
        title='Charged vertices'
    )
    charged_plaquettes: list[int] = Field(
        default_factory=list,
        description='Plaquette ids where static magnetic charges are to be located. Plaquettes must'
                    'be declared as inactive beforehand.',
        title='Charged plaquettes'
    )

    @field_validator('charged_vertices')
    @classmethod
    def check_num_vertices(cls, vlist: list[int]) -> list[int]:
        if len(vlist) % 2 != 0:
            raise ValueError('There must be an even number of charged vertices')
        return vlist

    def model_post_init(self, _):  # pylint: disable=arguments-differ
        if self.lattice == 'full_156q':
            self.lattice = '''
            *-*-*-*-*╷
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            * * * * *╎
             * * * * *
            *-*-*-*-*╵
            '''


class DMRGParameters(BaseModel):
    """Configuration for DMRG."""

    nsweeps: int = Field(
        default=5,
        description='Number of DMRG sweeps.',
        title='Number of sweeps'
    )
    maxdim: list[int] = Field(
        default=[10, 20, 100, 100, 200],
        description='Maximum size allowed for the bond dimension or rank of the MPS.',
        title='Maximum bond dimensions'
    )
    cutoff: float = Field(
        default=1.e-10,
        description='Truncation error cutoff or threshold to use for truncating the bond dimension'
                    ' or rank of the MPS.',
        title='Truncation error cutoff'
    )
    num_samples: int = Field(
        default=100_000,
        description='Number of times to sample the MPS to estimate the probability distribution of'
                    ' the ground state.',
        title='MPS sampling number'
    )
    julia_sysimage: Optional[str] = Field(
        default=None,
        description='Path to the precompiled ITensors sysimage.',
        title='ITensors.jl sysimage path'
    )


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    layout: Optional[list[int] | dict[tuple[str, int], int]] = Field(
        default=None,
        description='Qubit layout.',
        title='Qubit Layout'
    )
    sampled_steps: list[int] = Field(
        default=[2, 3],
        description="Trotter Evolution Steps to Sample From.",
        title="Sampled Trotter Steps",
    )
    dts: list[float] = Field(
        default=[0.25],
        description="Time Interval per Trotter step.",
        title="Time Intervals",
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )


class RuntimeParameters(BaseModel):
    """Configuration for runtime sampler."""

    config_block: str = Field(
        description='Name of the QiskitRuntimeConfig block name.',
        title='QiskitRuntimeConfig block'
    )
    shots_exp: int = Field(
        default=100_000,
        description='Shots per experiment circuit.',
        title='Shots (Experiment)'
    )
    shots_ref: int = Field(
        default=500_000,
        description='Shots per reference circuit.',
        title='Shots (Reference)'
    )


class CRBMParameters(BaseModel):
    """Configuration for conditional restricted Boltzmann machine used in configuration recovery."""

    num_h: int = Field(
        default=256,
        description='Number of hidden units of the restricted Boltzmann machine.',
        title='Number of Hidden Units',
        ge=1
    )
    train_batch_size: int = Field(
        default=64,
        description='Batch size for stochastic gradient descent.',
        title='Training Batch Size',
        ge=1
    )
    gen_batch_size: int = Field(
        default=10_000,
        description='Batch size for generation.',
        title='Generation Batch Size',
        ge=1
    )
    train_eval_unit: int = Field(
        default=100,
        description='The number of training steps to take the average of training metrics over.',
        title='Training Evaluation Unit',
        ge=1
    )
    learning_rate: float = Field(
        default=0.01,
        description='Learning rate for the Adam optimizer with weight decay',
        title='Learning Rate',
        gt=0.
    )
    weight_decay: float = Field(
        default=0.1,
        description='Weight decay coefficient (to be multiplied with LR) of adamw',
        title='Weight decay',
        gt=0.
    )
    num_epochs: int = Field(
        default=1000,
        description='Maximum number of epochs to train.',
        title='Maximum Epochs',
        ge=1
    )


class SKQDParameters(BaseModel):
    """Configuration for SKQD algorithm execution."""

    num_gen: int = Field(
        default=10,
        description="Number of bitstrings per sample to generate for the initial iteration of "
                    "configuration recovery.",
        title="Generated Samples",
        ge=1
    )
    pt_core_max: int = Field(
        default=6,
        description="Start with all states with plaquette excitations up to the given number.",
        title="Perturbative Core Excitations",
        ge=0
    )
    max_iterations: int = Field(
        default=10,
        description="Number of configuration recovery iterations.",
        title="Max Iteration",
        ge=0
    )
    energy_convergence: float = Field(
        default=1.e-7,
        description="Convergence condition (change in energy value between configuration recovery"
                    "iterations)",
        title="Energy Convergence",
        ge=0.
    )


class Parameters(BaseModel):
    """Workflow configuration parameters."""

    lgt: LGTParameters = Field(
        default_factory=LGTParameters,
        description='Lattice gauge theory definition.',
        title='Lattice Gauge Theory'
    )

    dmrg: Optional[DMRGParameters] = Field(
        default=None,
        description='DMRG and MPS sampling parameters.',
        title='DMRG'
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description='Trotter circuit parameters and transpiler settings.',
        title='Circuit'
    )

    runtime: RuntimeParameters = Field(
        default_factory=RuntimeParameters,
        description='Runtime configuration parameters.',
        title='Runtime'
    )

    crbm: CRBMParameters = Field(
        default_factory=CRBMParameters,
        description='Settings for conditional restricted Boltzmann machine.',
        title='CRBM'
    )

    skqd: SKQDParameters = Field(
        default_factory=SKQDParameters,
        description='Control of SKQD algorithm execution and solver parameters.',
        title='SKQD',
    )

    pkgpath: str = Field(
        description='Path of the directory tree where intermediate and final output are stored.',
        title='Output package name'
    )

    python_cmd: str = Field(
        default='python',
        description='Name of the HPC Command block for standard Python.',
        title='Python HPC Command block name'
    )

    python_cuda_cmd: str | None = Field(
        default=None,
        description='Name of the HPC Command block for CUDA-enabled Python'
                    ' (if different from standard).',
        title='Python (CUDA) HPC Command block name'
    )

    python_cpu_single_exec_profile: str = Field(
        default='python-cpu-single',
        description='Name of the Execution profile block for CPU-bound single-node Python job.',
        title='Python CPU-bound single-node execution profile'
    )

    python_gpu_single_exec_profile: str = Field(
        default='python-gpu-single',
        description='Name of the Execution profile block for GPU-bound single-node Python job.',
        title='Python GPU-bound single-node execution profile'
    )

    python_gpu_mpi_exec_profile: str = Field(
        default='python-gpu-mpi',
        description='Name of the Execution profile block for GPU-bound single-node Python job.',
        title='Python GPU-bound single-node execution profile'
    )

    hpc_profile: str = Field(
        default='local',
        description='Name of the HPC Profile block.',
        title='HPC Profile block name'
    )

    def model_post_init(self, _):  # pylint: disable=arguments-differ
        if self.python_cuda_cmd is None:
            self.python_cuda_cmd = self.python_cmd
