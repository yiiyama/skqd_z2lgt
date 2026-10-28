"""NumPy/CuPy dispatch proxy (same pattern as lp_ipeps.backend in the dissipative project)."""
from __future__ import annotations

import numpy as _numpy


class ArrayBackend:
    """Small NumPy/CuPy dispatch proxy.

    The DMRG modules import the singleton ``xp``. Selecting CuPy before the MPS/MPO are created
    moves every tensor, contraction (cuBLAS / cuTENSOR), SVD and eigendecomposition (cuSOLVER) to
    the requested CUDA device. The Hamiltonians handled here are real (Z, ZZ, X and Z-string terms
    only), so real dtypes are used throughout.
    """

    def __init__(self) -> None:
        self._module = _numpy
        self.name = "numpy"
        self.device = None
        self.dtype = _numpy.float64

    def configure(self, name: str = "auto", device: int = 0, precision: str = "double") -> str:
        name = name.lower()
        precision = precision.lower()
        if precision not in {"single", "double"}:
            raise ValueError("precision must be single or double")
        if name not in {"auto", "numpy", "cupy"}:
            raise ValueError("backend must be auto, numpy, or cupy")
        if name in {"auto", "cupy"}:
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA device found")
                cp.cuda.Device(device).use()
                try:
                    # Route tensordot/einsum through cuTENSOR when it is installed
                    from cupy._core import _accelerator
                    _accelerator.set_routine_accelerators(['cutensor', 'cub'])
                    _accelerator.set_reduction_accelerators(['cub'])
                except Exception:  # pylint: disable=broad-except
                    pass
                self._module = cp
                self.name = "cupy"
                self.device = int(device)
                self.dtype = cp.float32 if precision == "single" else cp.float64
                return self.name
            except Exception:
                if name == "cupy":
                    raise
        self._module = _numpy
        self.name = "numpy"
        self.device = None
        self.dtype = _numpy.float32 if precision == "single" else _numpy.float64
        return self.name

    def __getattr__(self, key):
        return getattr(self._module, key)

    def asarray_dev(self, value):
        """Move a host array to the active device with the working dtype."""
        return self._module.asarray(_numpy.asarray(value), dtype=self.dtype)

    def asnumpy(self, value):
        if self.name == "cupy":
            return self._module.asnumpy(value)
        return _numpy.asarray(value)

    def scalar(self, value):
        array = self.asnumpy(value)
        return array.item() if array.ndim == 0 else array

    def synchronize(self) -> None:
        if self.name == "cupy":
            self._module.cuda.Stream.null.synchronize()

    def memory_info(self) -> dict[str, int | str | None]:
        if self.name == "cupy":
            free, total = self._module.cuda.runtime.memGetInfo()
            used_pool = self._module.get_default_memory_pool().used_bytes()
            return {"backend": self.name, "device": self.device, "free": int(free),
                    "total": int(total), "pool_used": int(used_pool)}
        return {"backend": self.name, "device": None, "free": 0, "total": 0, "pool_used": 0}


xp = ArrayBackend()
