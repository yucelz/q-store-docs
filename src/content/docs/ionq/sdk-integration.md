---
title: IonQ SDK Integration
description: Integrating with IonQ's quantum computing SDK
---

Q-Store uses the official IonQ SDK (Cirq-IonQ) for quantum circuit execution.

## SDK Installation

```bash
pip install cirq cirq-ionq
```

## Basic Integration

```python
import cirq
import cirq_ionq as ionq

# Initialize IonQ service
service = ionq.Service(api_key=YOUR_KEY)

# Create circuit
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# Submit to IonQ
job = service.create_job(
    circuit=circuit,
    target='simulator',
    repetitions=1000
)

# Get results
results = job.results()
```

## Next Steps

- See [IonQ Hardware Selection](/ionq/hardware-selection)
- Check [IonQ Optimizations](/ionq/optimizations)
