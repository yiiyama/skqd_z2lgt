"""Functions dealing with the Qiskit Aer noise model."""
from collections.abc import Sequence
from qiskit_aer.noise import NoiseModel


def subset_noise_model(full_noise_model: NoiseModel, layout: Sequence[int]):
    """Construct a new NoiseModel containing errors only on physical qubits in the lay-in map."""
    layin_map = dict(zip(layout, range(len(layout))))
    subset_qubits = set(layout)

    noise_model = NoiseModel()
    for inst_name, error in full_noise_model._default_quantum_errors.items():
        noise_model.add_all_qubit_quantum_error(error, inst_name)

    for inst_name, error_dict in full_noise_model._local_quantum_errors.items():
        for qubits, error in error_dict.items():
            if not set(qubits) <= subset_qubits:
                continue
            logical_qubits = tuple(layin_map[q] for q in qubits)
            noise_model.add_quantum_error(error, inst_name, logical_qubits)

    if (error := full_noise_model._default_readout_error):
        noise_model.add_all_qubit_readout_error(error)

    for qubits, error in full_noise_model._local_readout_errors.items():
        if qubits[0] not in subset_qubits:
            continue
        logical_qubits = (layin_map[qubits[0]],)
        noise_model.add_readout_error(error, logical_qubits)

    return noise_model
