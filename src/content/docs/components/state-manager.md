---
title: State Manager
description: Core component for quantum state management in Q-Store v4.0.0
---

The **State Manager** is responsible for managing quantum states in the Q-Store v4.0.0 quantum-native database. It handles state encoding, superposition management, and quantum measurement operations.

## Overview

In Q-Store v4.0.0, the State Manager enables:
- **Superposition Storage**: Store vectors in multiple contexts simultaneously
- **Context-Aware Retrieval**: Query collapses superposition based on context
- **Quantum State Encoding**: Convert classical vectors to quantum amplitudes
- **Measurement Operations**: Extract classical results from quantum states

## Core Responsibilities

### Quantum State Encoding

The State Manager converts classical data into quantum states using two primary encoding methods:

#### Amplitude Encoding
Encodes N-dimensional classical vectors into log₂(N) qubits:
- **Input**: Classical vector `[v₀, v₁, ..., vₙ₋₁]`
- **Output**: Quantum state `|ψ⟩ = Σᵢ vᵢ|i⟩`
- **Benefit**: Exponential compression of data

#### Angle Encoding
Encodes classical data as rotation angles:
- **Input**: Classical vector `[x₀, x₁, ..., xₙ₋₁]`
- **Output**: Quantum state with rotations `Ry(xᵢ)|0⟩`
- **Benefit**: Direct feature mapping

### Superposition Management

Store vectors in multiple contexts using quantum superposition:

```python
from q_store import QuantumDatabase, DatabaseConfig

config = DatabaseConfig(
    enable_quantum=True,
    enable_superposition=True,
    pinecone_api_key="your-key"
)

db = QuantumDatabase(config)

# Store in superposition across multiple contexts
db.insert(
    id='doc_123',
    vector=embedding,
    contexts=[
        ('technical', 0.6),
        ('business', 0.3),
        ('legal', 0.1)
    ]
)
```

### Context-Aware Measurement

Query operations collapse superposition based on context:

```python
# Query with specific context - collapses to relevant state
results = db.query(
    vector=query_embedding,
    context='technical',
    top_k=10
)
# Returns results weighted toward technical context
```

## Key Methods

### create_superposition_state()

Creates quantum state in superposition of multiple contexts.

**Signature:**
```python
create_superposition_state(
    vector: np.ndarray,
    contexts: List[Tuple[str, float]],
    n_qubits: int
) -> QuantumCircuit
```

**Purpose:** Encode classical vector into quantum superposition for multi-context storage.

**Example:**
```python
circuit = state_manager.create_superposition_state(
    vector=embedding,
    contexts=[('tech', 0.7), ('business', 0.3)],
    n_qubits=8
)
```

### measure_with_context()

Collapses quantum state based on query context.

**Signature:**
```python
measure_with_context(
    state: QuantumCircuit,
    context: str,
    shots: int = 1000
) -> np.ndarray
```

**Purpose:** Extract classical results from quantum state using context-aware measurement.

**Example:**
```python
result = state_manager.measure_with_context(
    state=quantum_state,
    context='technical',
    shots=1000
)
```

### apply_decoherence()

Applies physics-based time-to-live mechanism.

**Signature:**
```python
apply_decoherence(
    time_delta: float,
    coherence_time: float
) -> List[str]
```

**Purpose:** Naturally expire old quantum states based on coherence time.

**Example:**
```python
# Cleanup states older than coherence time
expired_ids = state_manager.apply_decoherence(
    time_delta=1000.0,
    coherence_time=5000.0
)
```

## State Lifecycle

### 1. Creation
Classical vectors are encoded into quantum states:
- Normalize vector
- Determine qubit count (log₂N for amplitude encoding)
- Apply encoding circuit
- Store in quantum register

### 2. Storage
Quantum states are maintained with:
- **State ID**: Unique identifier
- **Contexts**: Superposition weights
- **Coherence Time**: Time until decoherence
- **Creation Timestamp**: For decoherence calculation

### 3. Retrieval
Context-aware measurement extracts results:
- Apply context-specific measurement basis
- Execute quantum circuit (shots=1000)
- Process measurement outcomes
- Return classical vector

### 4. Expiration
Physics-based decoherence removes stale states:
- Calculate time since creation
- Compare to coherence time
- Remove decohered states automatically

## Performance Characteristics

Based on v4.0.0 benchmarks:

| Operation | Time | Notes |
|-----------|------|-------|
| State creation | <1ms | Per quantum circuit |
| Encoding | ~59μs | Average gate operation |
| Measurement | ~0.03ms | 1000 shots |
| Decoherence check | <0.1ms | Per state |

## Integration with Other Components

### Circuit Builder
State Manager uses Circuit Builder for:
- Creating encoding circuits
- Building measurement operations
- Optimizing state preparation

### Classical Backend
State Manager coordinates with Pinecone for:
- Storing classical vector alongside quantum state
- Hybrid classical-quantum queries
- Fallback when quantum unavailable

### Entanglement Registry
State Manager enables entanglement via:
- Multi-qubit state preparation
- Correlated state management
- Relationship synchronization

## Configuration Options

```python
config = DatabaseConfig(
    # Quantum state management
    enable_quantum=True,
    enable_superposition=True,

    # Encoding settings
    encoding_method='amplitude',  # or 'angle'
    n_qubits=8,

    # Decoherence settings
    default_coherence_time=5000.0,  # milliseconds
    decoherence_check_interval=1000.0,

    # Backend settings
    quantum_sdk='ionq',  # or 'mock'
    quantum_target='simulator'
)
```

## Best Practices

### Choosing Encoding Method
- **Amplitude**: Use for dense, high-dimensional vectors (768D+)
- **Angle**: Use for sparse, low-dimensional features (<64D)

### Setting Coherence Time
- **Hot data**: 10000ms (10 seconds)
- **Normal data**: 5000ms (5 seconds)
- **Temporary data**: 1000ms (1 second)

### Context Management
- Limit to 3-5 contexts per vector
- Ensure context weights sum to 1.0
- Use meaningful context labels

## Limitations in v4.0.0

- **Qubit scaling**: Optimized for 4-8 qubits
- **Mock accuracy**: Random results in mock mode (~10-20%)
- **Real hardware**: Requires IonQ API key for actual quantum operations

## Next Steps

- Learn about [Circuit Builder](/components/circuit-builder) for quantum circuit generation
- Explore [Entanglement Registry](/components/entanglement-registry) for relationship management
- See [Quantum Principles](/concepts/quantum-principles) for theoretical foundation
