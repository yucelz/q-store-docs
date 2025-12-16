---
title: Quick Start
description: Get started with Q-Store in minutes
---

## Installation

```bash
# Latest version with v3.3 performance improvements
pip install q-store>=3.3.0
```

## Basic Setup

```python
from quantum_db import QuantumDatabase

# Initialize database
db = QuantumDatabase(
    classical_backend='pinecone',
    quantum_backend='ionq',
    ionq_api_key=YOUR_KEY
)
```

## Your First Query

### 1. Insert Data

```python
# Insert with superposition (multiple contexts)
db.insert(
    id='doc_123',
    vector=embedding,
    contexts=[
        ('normal_query', 0.7),      # 70% probability
        ('technical_query', 0.2),   # 20% probability
        ('historical_query', 0.1)   # 10% probability
    ],
    coherence_time=1000  # ms
)
```

### 2. Create Entangled Groups

```python
# Create entangled group (correlated entities)
db.create_entangled_group(
    group_id='related_docs',
    entity_ids=['doc_123', 'doc_124', 'doc_125'],
    correlation_strength=0.85
)
```

### 3. Query with Quantum Advantages

```python
# Query with context-aware collapse
results = db.query(
    vector=query_embedding,
    context='technical_query',     # Superposition collapses to this
    mode='balanced',                # Uncertainty tradeoff
    enable_tunneling=True,          # Find distant patterns
    top_k=10
)
```

### 4. Update with Auto-Propagation

```python
# Update entity (entangled partners auto-update)
db.update('doc_123', new_embedding)
# doc_124, doc_125 automatically reflect correlation
```

### 5. Adaptive Cleanup

```python
# Decoherence handles cleanup naturally
db.apply_decoherence()  # Old data naturally removed
```

## Configuration Options

```python
from quantum_db import QuantumDatabase, QuantumDatabaseConfig

config = QuantumDatabaseConfig(
    # Classical backend
    classical_backend='pinecone',
    classical_index_name='vectors',
    
    # Quantum backend
    quantum_backend='ionq',
    ionq_api_key=YOUR_KEY,
    n_qubits=20,
    target_device='simulator',  # or 'qpu.aria', 'qpu.forte'
    
    # Features
    enable_superposition=True,
    enable_entanglement=True,
    enable_decoherence=True,
    enable_tunneling=True,
)

db = QuantumDatabase(config)
```

## ML Training Quick Start (v3.3)

Train quantum neural networks with **50-100x performance improvements**:

```python
from q_store.core import (
    QuantumTrainer, QuantumModel, TrainingConfig, BackendManager
)

# Configure with v3.3 optimizations
config = TrainingConfig(
    pinecone_api_key="your-api-key",
    quantum_sdk="cirq",
    learning_rate=0.01,
    epochs=5,
    batch_size=10,
    n_qubits=8,
    
    # NEW v3.3 options for 24x speedup
    gradient_method='adaptive',      # Auto-selects best method
    enable_circuit_cache=True,       # 2-5x speedup
    enable_batch_execution=True,     # 5-10x speedup
    hardware_efficient_ansatz=True   # 33% fewer parameters
)

# Initialize and train
backend_manager = BackendManager(config)
backend = backend_manager.get_backend("cirq_ionq")

model = QuantumModel(input_dim=8, output_dim=2, n_layers=2, backend=backend)
trainer = QuantumTrainer(config, backend_manager)

# Train in minutes instead of hours
history = await trainer.train(model, data_loader)
print(f"Training completed in {history['total_time']:.1f}s")
```

**Performance**: 8-qubit model trains in ~8 minutes (vs 40 minutes in v3.2)

## Next Steps

- Learn about [v3.3 Performance Improvements](/getting-started/version-3-3)
- Understand [Quantum Principles](/concepts/quantum-principles)
- Explore [ML Training Performance](/advanced/ml-training-performance)
- Check [Domain Applications](/applications/ml-training)
- Read [Production Patterns](/production/connection-pooling) for deployment
