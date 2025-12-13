---
title: State Manager
description: Component responsible for quantum state management and superposition
---

The **State Manager** is the core component responsible for encoding classical vectors into quantum states, maintaining superposition, and executing measurements.

## Responsibilities

- Encode vectors into quantum states (amplitude encoding)
- Maintain superposition of multiple contexts
- Track coherence for each state
- Execute quantum measurements
- Decode measurement results

## Key Methods

### create_superposition()

Creates a quantum state in superposition of multiple contexts.

```python
create_superposition(
    vectors: List[Vector],
    contexts: List[str]
) -> QuantumState
```

**Example:**

```python
state = state_manager.create_superposition(
    vectors=[v1, v2, v3],
    contexts=['technical', 'general', 'historical']
)
```

### measure_with_context()

Collapses superposition based on query context.

```python
measure_with_context(
    state: QuantumState,
    query_context: str
) -> Vector
```

**Example:**

```python
# Superposition collapses to 'technical' context
result = state_manager.measure_with_context(
    state=quantum_state,
    query_context='technical'
)
```

### apply_decoherence()

Applies time-based decoherence to states.

```python
apply_decoherence(
    time_delta: float
) -> None
```

**Example:**

```python
# Clean up old states naturally
state_manager.apply_decoherence(
    time_delta=100  # milliseconds since last update
)
```

### get_coherent_states()

Returns all states still within coherence time.

```python
get_coherent_states() -> List[QuantumState]
```

## Amplitude Encoding

The state manager uses **amplitude encoding** to represent classical vectors as quantum states:

```
|ψ⟩ = Σᵢ αᵢ|i⟩

where:
- αᵢ = normalized vector components
- |αᵢ|² = probability of measuring state |i⟩
- Σᵢ |αᵢ|² = 1 (normalization)
```

### Implementation

```python
def amplitude_encode(vector: np.ndarray) -> cirq.Circuit:
    # Normalize vector
    normalized = vector / np.linalg.norm(vector)
    
    # Pad to power of 2
    n = len(normalized)
    n_qubits = int(np.ceil(np.log2(n)))
    padded = np.pad(normalized, (0, 2**n_qubits - n))
    
    # Create circuit
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Decompose into RY and CNOT gates
    circuit.append(decompose_to_ry_cnot(padded, qubits))
    
    return circuit
```

## Superposition Management

### State Representation

```python
@dataclass
class QuantumState:
    state_id: str
    qubits: List[cirq.Qid]
    contexts: Dict[str, float]  # context -> probability
    created_at: float
    coherence_time: float
    circuit: cirq.Circuit
```

### Context Weighting

```python
# Multiple contexts with probabilities
contexts = {
    'technical': 0.6,
    'general': 0.3,
    'historical': 0.1
}

# Sum must equal 1.0
assert sum(contexts.values()) == 1.0
```

## Coherence Tracking

### Coherence Model

```python
def is_coherent(state: QuantumState, current_time: float) -> bool:
    elapsed = current_time - state.created_at
    return elapsed < state.coherence_time
```

### Adaptive Coherence

Different data types get different coherence times:

```python
# Hot data - long coherence
hot_coherence = 5000  # 5 seconds

# Normal data - medium coherence  
normal_coherence = 1000  # 1 second

# Temporary data - short coherence
temp_coherence = 100  # 100 milliseconds
```

## Measurement Process

### Basis Selection

```python
# Computational basis (default)
basis = 'computational'

# Hadamard basis (for uncertainty control)
basis = 'hadamard'

# Custom basis
basis = custom_basis_matrix
```

### Measurement Execution

```python
def measure(
    state: QuantumState,
    context: str,
    basis: str = 'computational'
) -> MeasurementResult:
    # 1. Transform to measurement basis
    circuit = state.circuit.copy()
    if basis == 'hadamard':
        circuit.append([cirq.H(q) for q in state.qubits])
    
    # 2. Add measurement
    circuit.append(cirq.measure(*state.qubits, key='result'))
    
    # 3. Execute on quantum backend
    results = quantum_backend.execute(circuit)
    
    # 4. Decode results
    return decode_measurement(results, context)
```

## Performance Optimization

### State Caching

```python
class StateManager:
    def __init__(self):
        self.state_cache = LRUCache(maxsize=1000)
        
    def get_state(self, state_id: str) -> Optional[QuantumState]:
        return self.state_cache.get(state_id)
```

### Batch Encoding

```python
# Encode multiple vectors at once
states = state_manager.create_superposition_batch(
    vectors_list=[v1, v2, v3, ...],
    contexts_list=[ctx1, ctx2, ctx3, ...]
)
```

## Next Steps

- Learn about [Entanglement Registry](/components/entanglement-registry)
- Explore [Quantum Circuit Builder](/components/circuit-builder)
- See [IonQ Integration](/ionq/sdk-integration) for backend details
