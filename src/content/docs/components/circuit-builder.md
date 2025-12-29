---
title: Quantum Circuit Builder
description: Component for building and optimizing quantum circuits in Q-Store v4.0.0
---

The **Quantum Circuit Builder** generates hardware-agnostic quantum circuits for Q-Store v4.0.0, enabling quantum-enhanced database operations across multiple backend platforms.

## Overview

In Q-Store v4.0.0, the Circuit Builder provides:
- **Hardware Abstraction**: Build once, run on IonQ, Cirq, Qiskit, or mock backends
- **Optimized Circuits**: Gate-level optimization for minimal circuit depth
- **Verification**: Unit

arity checking and equivalence validation
- **PyTorch Integration**: Quantum circuits as trainable neural network layers

## Core Responsibilities

### Circuit Construction

The Circuit Builder creates quantum circuits for:

1. **Data Encoding**: Transform classical vectors into quantum states
2. **Variational Circuits**: Parameterized quantum layers for ML
3. **Measurement Operations**: Extract classical results from quantum states
4. **Entanglement Operations**: Multi-qubit correlation circuits

### Hardware Abstraction

Q-Store supports multiple quantum backends:

```python
from q_store import QuantumCircuit, DatabaseConfig

# Mock mode - development/testing (no API key needed)
config = DatabaseConfig(quantum_sdk='mock')

# IonQ simulator - realistic quantum simulation
config = DatabaseConfig(
    quantum_sdk='ionq',
    quantum_target='simulator',
    ionq_api_key="your-key"
)

# IonQ hardware - real quantum processor
config = DatabaseConfig(
    quantum_sdk='ionq',
    quantum_target='qpu',
    ionq_api_key="your-key"
)
```

### Circuit Optimization

All circuits are optimized for:
- **Minimal depth**: Reduce decoherence effects
- **Native gates**: Compile to hardware-specific gate sets
- **Unitarity**: Verify quantum mechanical validity

## Key Methods

### build_encoding_circuit()

Creates quantum circuit for classical data encoding.

**Signature:**
```python
build_encoding_circuit(
    vector: np.ndarray,
    encoding: str = 'amplitude',
    n_qubits: Optional[int] = None
) -> QuantumCircuit
```

**Parameters:**
- `vector`: Classical data to encode
- `encoding`: 'amplitude' or 'angle'
- `n_qubits`: Number of qubits (auto-calculated if None)

**Returns:** Quantum circuit encoding the input vector

**Example:**
```python
# Amplitude encoding: 768D vector → 10 qubits
circuit = builder.build_encoding_circuit(
    vector=embedding,
    encoding='amplitude',
    n_qubits=10
)
```

### build_variational_circuit()

Creates parameterized quantum circuit for ML.

**Signature:**
```python
build_variational_circuit(
    n_qubits: int,
    depth: int,
    parameters: np.ndarray,
    entanglement: str = 'linear'
) -> QuantumCircuit
```

**Parameters:**
- `n_qubits`: Number of qubits
- `depth`: Circuit depth (number of layers)
- `parameters`: Trainable rotation parameters
- `entanglement`: Pattern ('linear', 'circular', 'full')

**Returns:** Variational quantum circuit

**Example:**
```python
# 4-qubit, depth-2 variational circuit
circuit = builder.build_variational_circuit(
    n_qubits=4,
    depth=2,
    parameters=np.random.rand(24),  # 4 qubits × 3 gates × 2 layers
    entanglement='linear'
)
```

### build_measurement_circuit()

Adds measurement operations to circuit.

**Signature:**
```python
build_measurement_circuit(
    circuit: QuantumCircuit,
    basis: str = 'computational',
    measure_qubits: Optional[List[int]] = None
) -> QuantumCircuit
```

**Parameters:**
- `circuit`: Quantum circuit to measure
- `basis`: Measurement basis ('computational', 'hadamard', 'pauli_x/y/z')
- `measure_qubits`: Specific qubits to measure (all if None)

**Returns:** Circuit with measurement gates

**Example:**
```python
# Measure all qubits in computational basis
circuit_with_measurement = builder.build_measurement_circuit(
    circuit=quantum_circuit,
    basis='computational'
)
```

### optimize_circuit()

Optimizes circuit for target backend.

**Signature:**
```python
optimize_circuit(
    circuit: QuantumCircuit,
    backend: str,
    level: int = 2
) -> QuantumCircuit
```

**Parameters:**
- `circuit`: Circuit to optimize
- `backend`: Target backend ('ionq', 'cirq', 'qiskit', 'mock')
- `level`: Optimization level (0-3, higher = more aggressive)

**Returns:** Optimized quantum circuit

**Example:**
```python
# Optimize for IonQ hardware
optimized = builder.optimize_circuit(
    circuit=raw_circuit,
    backend='ionq',
    level=2
)
```

### verify_unitarity()

Verifies quantum circuit preserves unitarity.

**Signature:**
```python
verify_unitarity(
    circuit: QuantumCircuit,
    tolerance: float = 1e-10
) -> bool
```

**Parameters:**
- `circuit`: Circuit to verify
- `tolerance`: Numerical tolerance for unitarity check

**Returns:** True if circuit is unitary

**Example:**
```python
# Verify circuit is valid quantum operation
is_valid = builder.verify_unitarity(
    circuit=quantum_circuit,
    tolerance=1e-10
)
```

## Circuit Patterns

### 1. Encoding Circuit

```
Input: Classical vector [v₀, v₁, ..., vₙ]

Circuit:
┌─────────┐
│ Prepare │  Initialize |0⟩⊗ⁿ
└─────────┘
     │
┌─────────┐
│ Encode  │  Apply encoding gates (RY, RZ, etc.)
└─────────┘
     │
   Output: |ψ⟩ = Σᵢ vᵢ|i⟩
```

### 2. Variational Circuit

```
Input: Parameters θ = [θ₀, θ₁, ..., θₘ]

Circuit (per layer):
┌─────────────┐
│  Rotations  │  RY(θ), RZ(θ), RX(θ) on each qubit
└─────────────┘
       │
┌─────────────┐
│ Entanglement│  CNOT gates in pattern (linear/circular/full)
└─────────────┘

Repeat for 'depth' layers
```

### 3. Measurement Circuit

```
┌──────────┐
│  Basis   │  Optional basis rotation (H, S, etc.)
│ Rotation │
└──────────┘
     │
┌──────────┐
│  Measure │  Computational basis measurement
└──────────┘
     │
  Shots × 1000 → Classical bitstrings
```

## Performance Characteristics

Based on v4.0.0 benchmarks:

| Operation | Time | Qubits | Notes |
|-----------|------|--------|-------|
| Build encoding circuit | <1ms | 8 | Amplitude encoding |
| Build variational circuit | <1ms | 4-8 | Depth 2-4 |
| Gate operation | ~59μs | - | Average per gate |
| Circuit verification | <0.5ms | 8 | Unitarity check |
| Optimization | <5ms | 8 | Level 2 |

## PyTorch Integration

Q-Store v4.0.0 includes `QuantumLayer` for PyTorch:

```python
import torch
from q_store import QuantumLayer, DatabaseConfig

config = DatabaseConfig(quantum_sdk='mock')

# Create quantum layer
quantum_layer = QuantumLayer(
    n_qubits=4,
    depth=2,
    config=config
)

# Use in PyTorch model
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = torch.nn.Linear(10, 4)
        self.quantum = quantum_layer
        self.output = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)  # Quantum processing
        x = self.output(x)
        return x

model = HybridModel()
```

**Training Performance:**
- 500 samples, 2 epochs, 4 qubits: 19.5 seconds
- Full gradient computation via parameter shift rule
- CPU tensors (GPU acceleration pending)

## Backend-Specific Optimization

### IonQ Backend
- **Native Gates**: GPi, GPi2, MS (Mølmer-Sørensen)
- **Connectivity**: All-to-all (fully connected)
- **Optimization**: Compile to native gates, minimize MS gates

### Mock Backend
- **Purpose**: Development/testing without API keys
- **Accuracy**: Random results (~10-20%)
- **Speed**: Fastest option for development
- **Usage**: Set `quantum_sdk='mock'`

### Cirq/Qiskit Backends
- **Native Gates**: Backend-specific
- **Connectivity**: Varies by hardware
- **Optimization**: Automatic transpilation

## Circuit Verification

Q-Store performs multiple verification checks:

### 1. Unitarity Check
Verifies U†U = I (quantum operations must be reversible):
```python
is_unitary = builder.verify_unitarity(circuit)
```

### 2. Equivalence Validation
Confirms optimized circuit equals original:
```python
is_equivalent = builder.verify_equivalence(
    original_circuit,
    optimized_circuit
)
```

### 3. Parameter Count Validation
Ensures parameter count matches circuit structure:
```python
expected_params = n_qubits * 3 * depth
actual_params = len(parameters)
assert actual_params == expected_params
```

## Best Practices

### Choosing Circuit Depth
- **Depth 2**: Fast, minimal decoherence, lower expressivity
- **Depth 4**: Balanced (recommended for most use cases)
- **Depth 6+**: High expressivity, more decoherence, slower

### Encoding Selection
- **Amplitude**: Dense vectors (768D embeddings) → log₂(768) = 10 qubits
- **Angle**: Sparse features, direct mapping

### Entanglement Patterns
- **Linear**: Nearest-neighbor, minimal gates
- **Circular**: Ring topology, moderate connectivity
- **Full**: All-to-all, maximum expressivity, most gates

## Limitations in v4.0.0

- **Qubit Range**: Optimized for 4-8 qubits
- **Mock Mode**: Random accuracy (~10-20%), use for testing only
- **GPU Support**: Quantum layers return CPU tensors
- **Circuit Depth**: Deep circuits (>10 layers) require specialized acceleration

## Next Steps

- Learn about [State Manager](/components/state-manager) for quantum state management
- Explore [Entanglement Registry](/components/entanglement-registry) for multi-qubit operations
- See [Quantum Principles](/concepts/quantum-principles) for theoretical foundation
- Check [IonQ Integration](/ionq/overview) for hardware backend setup
