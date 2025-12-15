---
title: Quantum Circuit Builder
description: Component for building and optimizing quantum circuits for ML
---

The **Quantum Circuit Builder** generates optimized quantum circuits for hardware-agnostic execution, with enhanced support for ML training circuits in v3.2.

## Responsibilities

- Generate quantum circuits for multiple backends (Cirq, Qiskit)
- Implement data encoding circuits (amplitude, angle, basis)
- Create variational quantum circuits for ML (NEW in v3.2)
- Build entanglement operations with multiple patterns
- Handle measurement basis selection
- Optimize for native gate sets
- Build quantum gradient circuits (NEW in v3.2)

## Key Methods

### build_encoding_circuit()

Encodes classical vector as quantum amplitudes.

**Function Signature:**
```python
build_encoding_circuit(
    vector: np.ndarray,
    encoding_type: str = 'amplitude'
) -> QuantumCircuit
```

**Purpose:** Create quantum circuit that encodes classical data into quantum states.

**Encoding Types:**
- `amplitude`: Encode as state amplitudes
- `angle`: Encode as rotation angles
- `basis`: Encode as computational basis states

---

### build_variational_circuit()

Creates variational quantum circuit for ML (NEW in v3.2).

**Function Signature:**
```python
build_variational_circuit(
    n_qubits: int,
    depth: int,
    parameters: np.ndarray,
    entanglement: str = 'linear'
) -> QuantumCircuit
```

**Purpose:** Build parameterized quantum circuit with trainable rotation gates for quantum neural networks.

**Entanglement Patterns:**
- `linear`: Sequential CNOT chain
- `circular`: Ring topology with wrap-around
- `full`: All-to-all connectivity

---

### build_gradient_circuit()

Builds circuit for quantum gradient computation (NEW in v3.2).

**Function Signature:**
```python
build_gradient_circuit(
    base_circuit: QuantumCircuit,
    parameter_index: int,
    shift: float = np.pi / 2
) -> Tuple[QuantumCircuit, QuantumCircuit]
```

**Purpose:** Create forward and backward shifted circuits for parameter shift rule gradient computation.

---

### build_entanglement_circuit()

Creates entangled state for multiple vectors.

**Function Signature:**
```python
build_entanglement_circuit(
    n_qubits: int,
    entanglement_type: str = 'GHZ',
    pattern: str = 'linear'
) -> QuantumCircuit
```

**Purpose:** Generate entanglement operations between qubits using specified topology.

---

### build_tunneling_circuit()

Builds circuit for quantum tunneling search.

**Function Signature:**
```python
build_tunneling_circuit(
    source_state: np.ndarray,
    barrier_height: float,
    tunneling_strength: float = 0.5
) -> QuantumCircuit
```

**Purpose:** Create quantum circuit that enables tunneling through energy barriers for optimization.

---

### build_measurement_circuit()

Creates measurement in specified basis.

**Function Signature:**
```python
build_measurement_circuit(
    n_qubits: int,
    basis: str = 'computational',
    measure_indices: Optional[List[int]] = None
) -> QuantumCircuit
```

**Purpose:** Add measurement operations in chosen basis (computational, Hadamard, Pauli-X/Y/Z).

---

### optimize_circuit()

Optimizes circuit for target backend (NEW in v3.2).

**Function Signature:**
```python
optimize_circuit(
    circuit: QuantumCircuit,
    backend_type: str,
    optimization_level: int = 2
) -> QuantumCircuit
```

**Purpose:** Transpile and optimize circuit for specific quantum hardware native gates.

---

### decompose_to_native_gates()

Decomposes arbitrary gates to native gate set.

**Function Signature:**
```python
decompose_to_native_gates(
    circuit: QuantumCircuit,
    native_gates: List[str]
) -> QuantumCircuit
```

**Purpose:** Convert circuit to use only gates supported by target hardware.

---

### estimate_circuit_depth()

Estimates circuit depth for resource planning (NEW in v3.2).

**Function Signature:**
```python
estimate_circuit_depth(
    circuit: QuantumCircuit
) -> int
```

**Purpose:** Calculate circuit depth for coherence time and cost estimation.

---

### build_quantum_layer_circuit()

Builds complete quantum layer circuit for ML (NEW in v3.2).

**Function Signature:**
```python
build_quantum_layer_circuit(
    input_data: np.ndarray,
    parameters: np.ndarray,
    n_qubits: int,
    depth: int,
    entanglement: str = 'linear'
) -> QuantumCircuit
```

**Purpose:** Create full quantum neural network layer combining encoding, variational circuit, and measurement.

---

## Circuit Patterns for ML

### Variational Layer Structure

Each variational layer consists of:
1. **Rotation Layer**: RY, RZ, RX gates with trainable parameters
2. **Entanglement Layer**: CNOT gates in specified pattern
3. **Repeat**: For circuit depth

### Feature Map Circuits

Quantum feature maps for data encoding:
- **ZFeatureMap**: Single-qubit rotations
- **ZZFeatureMap**: Two-qubit interactions
- **PauliFeatureMap**: Multi-Pauli rotations

## Hardware Optimization

### Backend-Specific Optimization

Circuits are optimized for:
- **IonQ**: All-to-all connectivity, native GPi/GPi2/MS gates
- **IBM**: Limited connectivity, native RZ/SX/CNOT gates
- **Simulators**: No hardware constraints

### Gate Decomposition

Complex gates decomposed to:
- Single-qubit rotations (RX, RY, RZ)
- Two-qubit gates (CNOT, CZ)
- Native hardware gates

## Next Steps

- Learn about [State Manager](/components/state-manager)
- See [Tunneling Engine](/components/tunneling-engine)
- Explore [ML Model Training](/applications/ml-training)
