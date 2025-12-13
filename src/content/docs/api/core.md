---
title: Core API
description: Core Python API reference for Q-Store
---

## QuantumDatabase Class

Main interface for interacting with the quantum database.

### Initialization

```python
from quantum_db import QuantumDatabase, QuantumDatabaseConfig

# Simple initialization
db = QuantumDatabase(
    classical_backend='pinecone',
    quantum_backend='ionq',
    ionq_api_key=YOUR_KEY
)

# Advanced configuration
config = QuantumDatabaseConfig(
    classical_backend='pinecone',
    quantum_backend='ionq',
    ionq_api_key=YOUR_KEY,
    n_qubits=20,
    enable_superposition=True,
    enable_entanglement=True
)
db = QuantumDatabase(config)
```

### insert()

Insert a vector with optional superposition contexts.

```python
db.insert(
    id: str,
    vector: List[float],
    contexts: List[Tuple[str, float]] = None,
    coherence_time: float = 1000,
    metadata: Dict = None
) -> None
```

**Parameters:**
- `id`: Unique identifier for the vector
- `vector`: Embedding vector (any dimension)
- `contexts`: List of (context_name, probability) tuples
- `coherence_time`: Time in milliseconds before decoherence
- `metadata`: Optional metadata dictionary

**Example:**

```python
db.insert(
    id='doc_123',
    vector=[0.1, 0.2, 0.3, ...],
    contexts=[
        ('technical', 0.7),
        ('general', 0.3)
    ],
    coherence_time=5000,
    metadata={'source': 'arxiv', 'date': '2025-12-13'}
)
```

### query()

Query the database with quantum enhancements.

```python
db.query(
    vector: List[float],
    context: str = None,
    mode: str = 'balanced',
    enable_tunneling: bool = False,
    top_k: int = 10,
    classical_candidates: int = 1000
) -> List[QueryResult]
```

**Parameters:**
- `vector`: Query embedding vector
- `context`: Context for superposition collapse
- `mode`: `'precise'`, `'balanced'`, or `'exploratory'`
- `enable_tunneling`: Enable quantum tunneling search
- `top_k`: Number of results to return
- `classical_candidates`: Number of candidates from classical filter

**Returns:** List of `QueryResult` objects with fields:
- `id`: Result identifier
- `score`: Similarity score (0-1)
- `vector`: Result vector
- `metadata`: Associated metadata

**Example:**

```python
results = db.query(
    vector=query_embedding,
    context='technical',
    mode='balanced',
    enable_tunneling=True,
    top_k=10
)

for result in results:
    print(f"{result.id}: {result.score}")
```

### create_entangled_group()

Create an entangled group of related entities.

```python
db.create_entangled_group(
    group_id: str,
    entity_ids: List[str],
    correlation_strength: float = 0.8
) -> None
```

**Parameters:**
- `group_id`: Identifier for the entangled group
- `entity_ids`: List of entity IDs to entangle
- `correlation_strength`: Strength of correlation (0-1)

**Example:**

```python
db.create_entangled_group(
    group_id='tech_stocks',
    entity_ids=['AAPL', 'MSFT', 'GOOGL'],
    correlation_strength=0.85
)
```

### update()

Update an entity (propagates via entanglement).

```python
db.update(
    id: str,
    vector: List[float],
    metadata: Dict = None
) -> None
```

**Parameters:**
- `id`: Entity identifier
- `vector`: New embedding vector
- `metadata`: Updated metadata (optional)

**Example:**

```python
db.update(
    id='doc_123',
    vector=new_embedding,
    metadata={'updated_at': '2025-12-13'}
)
```

### delete()

Delete an entity and remove from entanglements.

```python
db.delete(id: str) -> None
```

### apply_decoherence()

Apply time-based decoherence cleanup.

```python
db.apply_decoherence() -> int
```

**Returns:** Number of states cleaned up

**Example:**

```python
cleaned = db.apply_decoherence()
print(f"Cleaned up {cleaned} decohered states")
```

### get_entangled_partners()

Get all entities entangled with a given entity.

```python
db.get_entangled_partners(id: str) -> List[str]
```

**Returns:** List of entangled entity IDs

## Configuration Class

### QuantumDatabaseConfig

```python
@dataclass
class QuantumDatabaseConfig:
    # Classical backend
    classical_backend: str = 'pinecone'
    classical_index_name: str = 'vectors'
    
    # Quantum backend
    quantum_backend: str = 'ionq'
    ionq_api_key: str = None
    n_qubits: int = 20
    target_device: str = 'simulator'
    
    # Superposition settings
    enable_superposition: bool = True
    max_contexts_per_vector: int = 5
    
    # Entanglement settings
    enable_entanglement: bool = True
    auto_detect_correlations: bool = True
    correlation_threshold: float = 0.7
    
    # Decoherence settings
    enable_decoherence: bool = True
    default_coherence_time: float = 1000.0
    
    # Tunneling settings
    enable_tunneling: bool = True
    tunnel_probability: float = 0.2
    barrier_threshold: float = 0.8
    
    # Performance settings
    quantum_batch_size: int = 100
    classical_candidate_pool: int = 1000
    cache_quantum_states: bool = True
```

## Result Classes

### QueryResult

```python
@dataclass
class QueryResult:
    id: str
    score: float
    vector: List[float]
    metadata: Dict
    quantum_enhanced: bool
    context_collapsed: str = None
```

### MeasurementResult

```python
@dataclass
class MeasurementResult:
    state_id: str
    collapsed_vector: List[float]
    context: str
    probability: float
    measurement_basis: str
```

## Exceptions

### QuantumBackendError

Raised when quantum backend operations fail.

```python
try:
    results = db.query(vector)
except QuantumBackendError as e:
    print(f"Quantum backend error: {e}")
    # Fallback to classical only
```

### CoherenceViolation

Raised when attempting to use decohered state.

```python
try:
    state = db.get_state(state_id)
except CoherenceViolation:
    print("State has decohered")
```

### EntanglementError

Raised when entanglement operations fail.

```python
try:
    db.create_entangled_group(group_id, entity_ids)
except EntanglementError as e:
    print(f"Entanglement failed: {e}")
```

## Next Steps

- See [REST API](/api/rest) for HTTP interface
- Check [Python SDK](/api/python-sdk) for async patterns
- Explore [Production Patterns](/production/connection-pooling)
