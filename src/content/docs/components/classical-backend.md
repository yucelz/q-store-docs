---
title: Classical Backend
description: Pinecone vector database integration for Q-Store v4.0.0 hybrid architecture
---

The **Classical Backend** provides persistent vector storage using Pinecone vector database, enabling Q-Store v4.0.0's hybrid classical-quantum architecture for scalable quantum-enhanced database operations.

## Overview

In Q-Store v4.0.0, the Classical Backend handles:
- **Persistent Storage**: Pinecone vector database for classical vectors
- **Hybrid Architecture**: Classical storage + quantum enhancement
- **Scalability**: Handle millions of vectors efficiently
- **Fallback**: Classical operations when quantum unavailable

## Core Architecture

Q-Store v4.0.0 uses Pinecone as the primary storage backend:

```
┌─────────────────────────────────────────┐
│         Q-Store v4.0.0 Database         │
└──────────────┬─────────────┬────────────┘
               │             │
       ┌───────┴─────┐  ┌────┴─────────┐
       │   Quantum   │  │  Classical   │
       │  Enhancement│  │   Storage    │
       │  (IonQ/Mock)│  │  (Pinecone)  │
       └─────────────┘  └──────────────┘
              │               │
              └───────┬───────┘
                      │
                Vector Operations
```

## Configuration

### Basic Setup

```python
from q_store import QuantumDatabase, DatabaseConfig

config = DatabaseConfig(
    # Classical backend (Pinecone)
    pinecone_api_key="your-pinecone-key",
    pinecone_environment="us-east-1",
    pinecone_index="quantum-vectors",

    # Quantum enhancement (optional)
    enable_quantum=True,
    quantum_sdk='ionq',  # or 'mock'
    ionq_api_key="your-ionq-key"  # if using IonQ
)

db = QuantumDatabase(config)
```

### Pinecone Index Setup

Q-Store requires minimum 768-dimensional embeddings:

```python
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="your-key",
    environment="us-east-1"
)

# Create index (if not exists)
if 'quantum-vectors' not in pinecone.list_indexes():
    pinecone.create_index(
        name='quantum-vectors',
        dimension=768,  # Minimum for Q-Store
        metric='cosine',
        pods=1,
        pod_type='p1.x1'
    )
```

## Key Methods

### insert()

Stores vector with optional quantum enhancement.

**Signature:**
```python
insert(
    id: str,
    vector: np.ndarray,
    metadata: Optional[Dict] = None,
    contexts: Optional[List[Tuple[str, float]]] = None
) -> None
```

**Parameters:**
- `id`: Unique vector ID
- `vector`: Embedding vector (768D+)
- `metadata`: Optional metadata
- `contexts`: Optional superposition contexts (quantum mode)

**Example:**
```python
# Classical insert
db.insert(
    id='doc_123',
    vector=embedding_768d,
    metadata={'title': 'Quantum Computing Basics'}
)

# Quantum superposition insert
db.insert(
    id='doc_456',
    vector=embedding_768d,
    contexts=[
        ('technical', 0.7),
        ('business', 0.3)
    ]
)
```

### query()

Searches for similar vectors.

**Signature:**
```python
query(
    vector: np.ndarray,
    top_k: int = 10,
    filter: Optional[Dict] = None,
    context: Optional[str] = None,
    enable_tunneling: bool = False
) -> List[QueryResult]
```

**Parameters:**
- `vector`: Query embedding
- `top_k`: Number of results
- `filter`: Metadata filter
- `context`: Quantum context (if using superposition)
- `enable_tunneling`: Enable quantum tunneling

**Example:**
```python
# Classical query
results = db.query(
    vector=query_embedding,
    top_k=10,
    filter={'category': 'tech'}
)

# Quantum-enhanced query
results = db.query(
    vector=query_embedding,
    top_k=10,
    context='technical',  # Collapse superposition
    enable_tunneling=True  # Find globally best matches
)
```

### update()

Updates existing vector.

**Signature:**
```python
update(
    id: str,
    vector: np.ndarray,
    metadata: Optional[Dict] = None
) -> None
```

**Example:**
```python
db.update(
    id='doc_123',
    vector=updated_embedding,
    metadata={'updated_at': '2024-12-29'}
)
```

### delete()

Removes vector from database.

**Signature:**
```python
delete(
    id: str
) -> None
```

**Example:**
```python
db.delete('doc_123')
```

### fetch()

Retrieves vector by ID.

**Signature:**
```python
fetch(
    id: str
) -> Optional[Vector]
```

**Example:**
```python
vector = db.fetch('doc_123')
```

## Hybrid Classical-Quantum Operations

### Automatic Fallback

Q-Store automatically falls back to classical when quantum unavailable:

```python
# If quantum backend unavailable, uses classical Pinecone
db = QuantumDatabase(config)

results = db.query(vector, top_k=10)
# Uses quantum if available, falls back to classical
```

### Selective Quantum Enhancement

Enable quantum features selectively:

```python
# Classical insert (fast, no quantum overhead)
db.insert(id='doc1', vector=v1, enable_quantum=False)

# Quantum insert (with superposition)
db.insert(
    id='doc2',
    vector=v2,
    contexts=[('tech', 0.7), ('business', 0.3)],
    enable_quantum=True
)
```

### Performance Optimization

Use classical for high-volume, quantum for high-value:

```python
# Bulk insert - use classical
for i, vector in enumerate(bulk_vectors):
    db.insert(f'doc_{i}', vector, enable_quantum=False)

# Important query - use quantum
critical_results = db.query(
    vector=important_query,
    enable_tunneling=True,  # Quantum tunneling
    context='critical'       # Quantum superposition
)
```

## Performance Characteristics

| Operation | Classical (Pinecone) | Quantum Enhanced | Notes |
|-----------|----------------------|------------------|-------|
| Insert | <10ms | <15ms | +5ms for quantum |
| Query | ~30ms | ~60-100ms | +30-70ms for quantum |
| Update | <10ms | <15ms | +5ms for quantum |
| Delete | <5ms | <5ms | No quantum overhead |
| Fetch | <5ms | <5ms | No quantum overhead |

## Metadata Filtering

Pinecone supports metadata filtering:

```python
# Insert with metadata
db.insert(
    id='doc_123',
    vector=embedding,
    metadata={
        'category': 'tech',
        'year': 2024,
        'author': 'Alice'
    }
)

# Query with filter
results = db.query(
    vector=query_embedding,
    top_k=10,
    filter={
        'category': {'$eq': 'tech'},
        'year': {'$gte': 2020}
    }
)
```

## Best Practices

### Choosing Vector Dimensions

- **768D**: Minimum for Q-Store (BERT embeddings)
- **1024D**: Better for complex semantics
- **1536D**: OpenAI embeddings (ada-002)
- **Higher**: More expensive, ensure Pinecone pod supports it

### Index Configuration

```python
# Production configuration
pinecone.create_index(
    name='quantum-vectors-prod',
    dimension=768,
    metric='cosine',
    pods=2,  # Scale for throughput
    pod_type='p1.x2',  # Higher performance
    metadata_config={
        'indexed': ['category', 'year']  # Index frequently filtered fields
    }
)
```

### Batch Operations

For efficiency, batch inserts:

```python
# Batch insert (more efficient)
vectors = [(f'doc_{i}', embedding) for i, embedding in enumerate(embeddings)]
db.batch_insert(vectors, batch_size=100)
```

## Limitations in v4.0.0

- **Pinecone Required**: No alternative classical backends in v4.0
- **Minimum 768D**: Lower-dimensional vectors not supported
- **Quantum Overhead**: +30-70ms per quantum operation
- **Cost**: Both Pinecone and IonQ/quantum costs

## Monitoring and Debugging

### Check Database Stats

```python
stats = db.get_stats()
print(f"Total vectors: {stats['vector_count']}")
print(f"Index dimension: {stats['dimension']}")
print(f"Quantum enabled: {stats['quantum_enabled']}")
```

### Verify Pinecone Connection

```python
# Test Pinecone connection
try:
    db.fetch('test_id')
    print("Pinecone connection: OK")
except Exception as e:
    print(f"Pinecone error: {e}")
```

### Monitor Quantum Usage

```python
# Check quantum vs classical usage
stats = db.get_quantum_stats()
print(f"Quantum queries: {stats['quantum_queries']}")
print(f"Classical fallbacks: {stats['classical_fallbacks']}")
print(f"Quantum success rate: {stats['quantum_success_rate']:.1%}")
```

## Migration from Other Databases

Q-Store v4.0.0 supports migration from other vector databases:

```python
# Example: Migrate from existing Pinecone index
source_index = pinecone.Index('old-index')
target_db = QuantumDatabase(config)

# Fetch all vectors
vectors = source_index.fetch(ids=all_ids)

# Insert into Q-Store
for id, data in vectors.items():
    target_db.insert(
        id=id,
        vector=data['values'],
        metadata=data.get('metadata')
    )
```

## Next Steps

- Learn about [State Manager](/components/state-manager) for quantum operations
- Explore [Entanglement Registry](/components/entanglement-registry) for relationships
- See [IonQ Integration](/ionq/overview) for quantum backend setup
- Check [Quick Start](/getting-started/quick-start) for complete examples
