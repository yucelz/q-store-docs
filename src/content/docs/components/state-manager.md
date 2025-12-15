---
title: State Manager
description: Component responsible for quantum state management and ML model states
---

The **State Manager** is the core component responsible for encoding classical data into quantum states, maintaining superposition, and managing ML model parameters in v3.2.

## Responsibilities

- Encode classical data into quantum states (amplitude and angle encoding)
- Maintain quantum superposition for ML features
- Track and manage ML model parameter states
- Execute quantum measurements
- Handle state persistence and checkpointing
- Manage coherence and decoherence for quantum states

## Key Methods

### create_superposition()

Creates a quantum state in superposition of multiple contexts.

**Function Signature:**
```python
create_superposition(
    state_id: str,
    vectors: List[np.ndarray],
    contexts: List[str],
    coherence_time: float = 1000.0
) -> QuantumState
```

**Purpose:** Encode classical vectors into quantum superposition states for context-aware processing.

---

### create_model_state()

Creates a quantum state for ML model parameters (NEW in v3.2).

**Function Signature:**
```python
create_model_state(
    model_id: str,
    parameters: np.ndarray,
    architecture: Dict[str, Any],
    metadata: Optional[Dict] = None
) -> QuantumState
```

**Purpose:** Store trainable quantum circuit parameters as persistent quantum states.

---

### update_model_parameters()

Updates ML model parameters in quantum state (NEW in v3.2).

**Function Signature:**
```python
update_model_parameters(
    model_id: str,
    new_parameters: np.ndarray
) -> None
```

**Purpose:** Update quantum circuit parameters during training iterations.

---

### measure_with_context()

Collapses superposition based on query context.

**Function Signature:**
```python
measure_with_context(
    state: QuantumState,
    query_context: str,
    shots: int = 1000
) -> np.ndarray
```

**Purpose:** Execute context-aware measurement to extract classical results from quantum states.

---

### apply_decoherence()

Applies time-based decoherence to states.

**Function Signature:**
```python
apply_decoherence(
    time_delta: float
) -> None
```

**Purpose:** Naturally expire old quantum states based on coherence time limits.

---

### get_coherent_states()

Returns all states still within coherence time.

**Function Signature:**
```python
get_coherent_states() -> List[QuantumState]
```

**Purpose:** Retrieve active quantum states that haven't decohered.

---

### save_state()

Persists quantum state to storage (NEW in v3.2).

**Function Signature:**
```python
save_state(
    state: QuantumState,
    storage_path: str
) -> None
```

**Purpose:** Save quantum state and parameters for model checkpointing.

---

### load_state()

Restores quantum state from storage (NEW in v3.2).

**Function Signature:**
```python
load_state(
    storage_path: str
) -> QuantumState
```

**Purpose:** Load previously saved quantum states for model resumption.

---

## State Encoding Methods

### amplitude_encode()

Encodes classical data as quantum amplitudes.

**Function Signature:**
```python
amplitude_encode(
    data: np.ndarray,
    n_qubits: Optional[int] = None
) -> QuantumCircuit
```

**Purpose:** Encode data vector as quantum state amplitudes: |ψ⟩ = Σᵢ αᵢ|i⟩

---

### angle_encode()

Encodes classical data as rotation angles.

**Function Signature:**
```python
angle_encode(
    data: np.ndarray,
    n_qubits: int
) -> QuantumCircuit
```

**Purpose:** Encode data as rotation angles: Ry(θᵢ)|0⟩ where θᵢ ∝ xᵢ

---

## State Management

### State Representation

Quantum states are represented with the following attributes:
- `state_id`: Unique identifier
- `parameters`: Quantum circuit parameters (trainable)
- `contexts`: Context weights for superposition
- `created_at`: Timestamp
- `coherence_time`: Time until decoherence
- `metadata`: Additional state information

### Coherence Tracking

States are tracked with coherence times:
- **ML Model States**: Persistent (infinite coherence)
- **Hot Data**: 5000ms coherence
- **Normal Data**: 1000ms coherence  
- **Temporary Data**: 100ms coherence

## Next Steps

- Learn about [Entanglement Registry](/components/entanglement-registry)
- Explore [Quantum Circuit Builder](/components/circuit-builder)
- See [ML Model Training](/applications/ml-training) for training workflows
