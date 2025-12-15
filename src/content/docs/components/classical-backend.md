---
title: Classical Backend
description: Integration with classical vector databases and training data storage
---

The **Classical Backend** handles bulk storage, training data management, and model checkpointing using mature vector databases in v3.2.

## Supported Backends

- **Pinecone**: Managed vector database (primary for v3.2)
- **pgvector**: PostgreSQL extension
- **Qdrant**: Open-source vector database
- **Redis**: Caching and temporary storage

## Responsibilities

- Store large-scale training datasets
- Manage model checkpoints and metadata
- Provide efficient batch data loading
- Handle embeddings and feature vectors
- Cache frequently accessed data
- Support distributed storage for ML training (NEW in v3.2)

## Key Methods

### store_training_data()

Stores training dataset in classical storage (NEW in v3.2).

**Function Signature:**
```python
store_training_data(
    dataset_id: str,
    data: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[Dict] = None
) -> None
```

**Purpose:** Persist training data for ML workflows with associated labels and metadata.

---

### load_training_batch()

Loads training batch from storage (NEW in v3.2).

**Function Signature:**
```python
load_training_batch(
    dataset_id: str,
    batch_size: int,
    shuffle: bool = True,
    offset: int = 0
) -> Tuple[np.ndarray, np.ndarray]
```

**Purpose:** Efficiently load batches of training data for iterative training.

---

### save_checkpoint()

Saves model checkpoint (NEW in v3.2).

**Function Signature:**
```python
save_checkpoint(
    model_id: str,
    epoch: int,
    parameters: np.ndarray,
    metrics: Dict[str, float],
    metadata: Optional[Dict] = None
) -> str
```

**Purpose:** Persist model state for resuming training or inference.

---

### load_checkpoint()

Loads model checkpoint (NEW in v3.2).

**Function Signature:**
```python
load_checkpoint(
    checkpoint_path: str
) -> Dict[str, Any]
```

**Purpose:** Restore model parameters and training state from checkpoint.

---

### create_data_loader()

Creates data loader for training (NEW in v3.2).

**Function Signature:**
```python
create_data_loader(
    dataset_id: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader
```

**Purpose:** Generate efficient async data loader for training loops.

---

### insert_vector()

Inserts single vector with metadata.

**Function Signature:**
```python
insert_vector(
    id: str,
    vector: np.ndarray,
    metadata: Optional[Dict] = None
) -> None
```

**Purpose:** Store individual embeddings or feature vectors.

---

### batch_insert()

Inserts multiple vectors efficiently.

**Function Signature:**
```python
batch_insert(
    ids: List[str],
    vectors: List[np.ndarray],
    metadata: Optional[List[Dict]] = None
) -> None
```

**Purpose:** Bulk insert operation for efficient data ingestion.

---

### query()

Queries for similar vectors.

**Function Signature:**
```python
query(
    query_vector: np.ndarray,
    top_k: int = 10,
    filter: Optional[Dict] = None
) -> List[QueryResult]
```

**Purpose:** Retrieve nearest neighbors for similarity search.

---

### update_vector()

Updates existing vector.

**Function Signature:**
```python
update_vector(
    id: str,
    vector: np.ndarray,
    metadata: Optional[Dict] = None
) -> None
```

**Purpose:** Modify stored vectors and associated metadata.

---

### delete_vector()

Deletes vector by ID.

**Function Signature:**
```python
delete_vector(
    id: str
) -> None
```

**Purpose:** Remove vectors from storage.

---

### get_statistics()

Returns storage statistics.

**Function Signature:**
```python
get_statistics() -> Dict[str, Any]
```

**Purpose:** Monitor storage usage, dataset sizes, and performance metrics.

---

## Configuration Examples

### Pinecone Configuration (Recommended for v3.2)

```python
backend_config = {
    'type': 'pinecone',
    'api_key': PINECONE_API_KEY,
    'environment': 'us-east-1',
    'index_name': 'quantum-ml-training',
    'dimension': 768,
    'metric': 'cosine'
}
```

### pgvector Configuration

```python
backend_config = {
    'type': 'pgvector',
    'host': 'localhost',
    'port': 5432,
    'database': 'quantum_db',
    'user': 'postgres',
    'password': 'password',
    'table_name': 'embeddings'
}
```

### Qdrant Configuration

```python
backend_config = {
    'type': 'qdrant',
    'host': 'localhost',
    'port': 6333,
    'collection_name': 'quantum-vectors',
    'dimension': 768
}
```

### Redis Configuration (Caching)

```python
backend_config = {
    'type': 'redis',
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'ttl': 3600  # 1 hour cache
}
```

## ML Training Integration (v3.2)

### Training Data Management

Classical backends efficiently manage:
- Large training datasets (millions of samples)
- Feature embeddings and labels
- Training/validation/test splits
- Data augmentation caches

### Model Persistence

Checkpoint storage includes:
- Model parameters (quantum circuit angles)
- Optimizer states
- Training metrics and history
- Architecture configurations

### Distributed Training

Support for distributed ML:
- Shared dataset access across workers
- Synchronized checkpoint storage
- Distributed batch loading
- Training progress tracking

## Performance Optimization

### Batch Operations

Use batch operations for efficiency:
- `batch_insert()` for data ingestion
- Batch loading for training
- Parallel query execution

### Caching Strategy

Multi-level caching:
- Redis for hot data
- Pinecone for persistent storage
- Local memory cache for active batches

### Index Optimization

Vector index tuning:
- Dimension reduction where appropriate
- Metric selection (cosine, euclidean, dot product)
- Quantization for large-scale storage

## Next Steps

- See [State Manager](/components/state-manager) for quantum integration
- Learn about [ML Model Training](/applications/ml-training)
- Explore [Deployment](/deployment/cloud) options
