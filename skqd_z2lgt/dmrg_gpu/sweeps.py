"""Sweep schedule modeled on ITensors' ``Sweeps``.

Each parameter is a per-sweep list; a list shorter than ``nsweeps`` repeats its last entry, as with
``setmaxdim!(sweeps, 10, 20, 100)`` in ITensors. A schedule can also be given as a table

    maxdim  mindim  cutoff  noise
    20      10      1e-14   1e-5
    100     20      1e-14   1e-6
    200     20      1e-14   0

where the last row repeats for the remaining sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


PARAMETERS = {
    'maxdim': int,
    'mindim': int,
    'cutoff': float,
    'noise': float,
    'krylovdim': int,
    'maxiter': int,
}


@dataclass
class SweepParameters:
    """Parameters of a single sweep."""
    maxdim: int
    mindim: int
    cutoff: float
    noise: float
    krylovdim: int
    maxiter: int


@dataclass
class Sweeps:
    """DMRG sweep schedule (one sweep = left->right + right->left).

    Attributes:
        nsweeps: Number of sweeps.
        maxdim: Maximum bond dimension.
        mindim: Minimum bond dimension kept regardless of the cutoff (when available).
        cutoff: Maximum discarded weight per truncation.
        noise: White's density-matrix perturbation strength.
        krylovdim: Lanczos Krylov-space dimension of the local eigensolver.
        maxiter: Lanczos restarts of the local eigensolver.
    """
    nsweeps: int
    maxdim: list[int] = field(default_factory=lambda: [256])
    mindim: list[int] = field(default_factory=lambda: [1])
    cutoff: list[float] = field(default_factory=lambda: [1.e-10])
    noise: list[float] = field(default_factory=lambda: [0.])
    krylovdim: list[int] = field(default_factory=lambda: [20])
    maxiter: list[int] = field(default_factory=lambda: [2])

    def __post_init__(self):
        if self.nsweeps < 1:
            raise ValueError('nsweeps must be positive')
        for name, cast in PARAMETERS.items():
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                value = [value]
            if len(value) == 0:
                raise ValueError(f'Empty sweep parameter {name}')
            setattr(self, name, [cast(v) for v in value])

    def __len__(self) -> int:
        return self.nsweeps

    def __getitem__(self, isweep: int) -> SweepParameters:
        if not 0 <= isweep < self.nsweeps:
            raise IndexError(isweep)
        return SweepParameters(**{name: _entry(getattr(self, name), isweep)
                                  for name in PARAMETERS})

    def __iter__(self):
        return (self[isweep] for isweep in range(self.nsweeps))

    @classmethod
    def from_table(cls, table: str, nsweeps: Optional[int] = None, **defaults) -> 'Sweeps':
        """Build from a whitespace-separated table with a header row of parameter names."""
        rows = [line.split() for line in table.strip().splitlines()
                if line.strip() and not line.strip().startswith('#')]
        header, rows = rows[0], rows[1:]
        if unknown := set(header) - set(PARAMETERS):
            raise ValueError(f'Unknown sweep parameter(s) {unknown}')
        columns = {name: [row[icol] for row in rows] for icol, name in enumerate(header)}
        return cls(nsweeps=nsweeps or len(rows), **defaults, **columns)

    @classmethod
    def from_config(cls, conf: Any) -> 'Sweeps':
        """Build from a Sweeps, a table string, or a dict (e.g. a yaml section) of the fields.

        A dict may contain a 'table' entry; other entries are defaults / overrides. Without
        nsweeps, the length of the longest list is used.
        """
        if isinstance(conf, Sweeps):
            return conf
        if isinstance(conf, str):
            return cls.from_table(conf)
        conf = dict(conf)
        if 'table' in conf:
            return cls.from_table(conf.pop('table'), **conf)
        if 'nsweeps' not in conf:
            lengths = [len(v) for k, v in conf.items() if k in PARAMETERS and isinstance(v, list)]
            conf['nsweeps'] = max(lengths, default=1)
        return cls(**conf)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self) -> str:
        lines = ['sweep ' + ' '.join(f'{name:>9}' for name in PARAMETERS)]
        for isweep, params in enumerate(self):
            lines.append(f'{isweep:5d} ' + ' '.join(f'{getattr(params, name):>9.3g}'
                                                     for name in PARAMETERS))
        return '\n'.join(lines)


def _entry(values: list, isweep: int):
    return values[min(isweep, len(values) - 1)]
