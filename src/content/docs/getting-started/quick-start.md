---
title: Quick Start
description: Get started with Q-Store v4.1.1 in 5 minutes
---

Get up and running with Q-Store v4.1.1 with async quantum execution - no API keys required for development!

:::tip[New in v4.1.1]
- ⚡ **Async quantum execution**: 10-20× faster than v4.0
- 🎆 **Quantum-first architecture**: 70% quantum computation
- 💾 **Zero-blocking storage**: Async Zarr + Parquet
- 🔧 **Fixed PyTorch integration**: Full autograd support
:::

## Installation

```bash
# Latest version with async execution and quantum-first architecture
pip install q-store==4.1.1

# With async support (recommended)
pip install q-store[async]==4.1.1

# Full installation (all backends)
pip install q-store[all]==4.1.1
```

**Requirements:** Python 3.11+

## Your First Quantum Circuit

```python
from q_store import QuantumCircuit

# Create a simple quantum circuit
circuit = QuantumCircuit(n_qubits=2)
circuit.h(0)        # Hadamard gate on qubit 0
circuit.cnot(0, 1)  # CNOT gate (control=0, target=1)

# Simulate the circuit
result = circuit.simulate()
print(result)
```

**Run it:**
```bash
python examples/basic_usage.py
```

## Hybrid Quantum-Classical ML (v4.1.1)

### Quantum-First Architecture (NEW!)

```python
import asyncio
from q_store.layers import (
    QuantumFeatureExtractor,
    QuantumPooling,
    QuantumReadout
)
from q_store.runtime import AsyncQuantumExecutor

# Build quantum-first model (70% quantum!)
model = Sequential([
    Flatten(),
    QuantumFeatureExtractor(n_qubits=8, depth=4, backend='ionq'),
    QuantumPooling(n_qubits=4),
    QuantumFeatureExtractor(n_qubits=4, depth=3),
    QuantumReadout(n_qubits=4, n_classes=10)
])

# Async training loop (non-blocking!)
async def train_model():
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            # Async forward pass (never blocks!)
            predictions = await model.forward_async(batch_x)
            
            loss = criterion(predictions, batch_y)
            gradients = await model.backward_async(loss)
            optimizer.step(gradients)
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Run async training
asyncio.run(train_model())
```

### PyTorch Integration (Fixed in v4.1.1!)

```python
import torch
import torch.nn as nn
from q_store.torch import QuantumLayer

# Build hybrid model
model = nn.Sequential(
    nn.Linear(784, 16),
    QuantumLayer(n_qubits=8, depth=4, backend='ionq'),  # Now with async!
    nn.Linear(24, 10)  # 8 qubits × 3 bases = 24 features
)

# Train like any PyTorch model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    output = model(input_data)
    loss = criterion(output, labels)
    loss.backward()  # Quantum gradients via SPSA
    optimizer.step()
```

**Run it:**
```bash
# Mock mode (instant, free, for development)
python examples/pytorch/fashion_mnist.py --samples 500 --epochs 2

# IonQ Simulator (real API, free, ~38 min for 1K images)
python examples/pytorch/cats_vs_dogs.py --no-mock --samples 1000 --epochs 5

# Note: Classical GPU training is 183-457× faster for production
# Use quantum for: research, small datasets, algorithm development
```

## Quantum-Enhanced Database

### Basic Setup

```python
from q_store import QuantumDatabase, DatabaseConfig
from q_store.runtime import AsyncQuantumExecutor

# Mock mode - no API keys needed
config = DatabaseConfig(
    quantum_sdk='mock',  # Use mock quantum backend
    enable_quantum=True,
    enable_superposition=True,
    max_concurrent=100,  # NEW in v4.1: async execution
    batch_size=20        # NEW in v4.1: circuit batching
)

db = QuantumDatabase(config)
```

### 1. Insert with Superposition

Store vectors in multiple contexts simultaneously:

```python
import numpy as np

# Store document in superposition
db.insert(
    id='doc_123',
    vector=np.random.rand(768),  # Your embedding
    contexts=[
        ('technical', 0.7),   # 70% weight
        ('business', 0.2),    # 20% weight
        ('legal', 0.1)        # 10% weight
    ]
)
```

### 2. Context-Aware Queries

Query collapses superposition to specific context:

```python
# Query for technical context
results = db.query(
    vector=query_embedding,
    context='technical',  # Collapses to technical context
    top_k=10
)
```

### 3. Entangle Related Entities

Create automatic correlation updates:

```python
# Entangle related documents
db.create_entangled_group(
    group_id='tech_docs',
    entity_ids=['doc_1', 'doc_2', 'doc_3'],
    correlation_strength=0.85
)

# Update one - others automatically adjust
db.update('doc_1', new_embedding)
# doc_2 and doc_3 automatically reflect correlation!
```

### 4. Quantum Tunneling Search

Find globally optimal matches:

```python
# Classical search (local optimum)
classical_results = db.query(
    vector=query,
    enable_tunneling=False,
    top_k=10
)

# Quantum tunneling (global search)
quantum_results = db.tunnel_search(
    query=query,
    barrier_threshold=0.7,
    tunneling_strength=0.6,
    top_k=10
)
# Finds patterns classical search misses!
```

### 5. Automatic Decoherence

Physics-based time-to-live:

```python
# Recent data - long coherence
db.insert(
    id='recent_doc',
    vector=embedding,
    coherence_time=86400000  # 24 hours
)

# Old data - natural decay
db.insert(
    id='old_doc',
    vector=old_embedding,
    coherence_time=3600000  # 1 hour
)

# Cleanup happens automatically
db.apply_decoherence()
```

### 6. Zero-Blocking Storage (NEW in v4.1.1!)

Async checkpoints and metrics:

```python
from q_store.storage import AsyncMetricsLogger, CheckpointManager

# Async metrics (never blocks training!)
metrics = AsyncMetricsLogger('experiments/run_001/metrics.parquet')
await metrics.log({
    'epoch': 1,
    'loss': 0.342,
    'circuit_time_ms': 107,
    'cost_usd': 0.0
})

# Async checkpoints (compressed Zarr)
checkpoints = CheckpointManager('experiments/run_001/checkpoints')
await checkpoints.save(
    epoch=10,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)
```

## Production Setup (Optional)

For production with real quantum backend and persistent storage:

### 1. Create `.env` File

```env
IONQ_API_KEY=your_ionq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
```

### 2. Configure Database

```python
from q_store import QuantumDatabase, DatabaseConfig

config = DatabaseConfig(
    # Quantum backend
    quantum_sdk='ionq',
    ionq_api_key='your-ionq-key',
    quantum_target='simulator',  # or 'qpu' for real hardware

    # Classical storage
    pinecone_api_key='your-pinecone-key',
    pinecone_environment='us-east-1',
    pinecone_index='quantum-vectors',

    # Features
    enable_quantum=True,
    enable_superposition=True,
    enable_entanglement=True,
    enable_tunneling=True,

    # Performance
    n_qubits=8,
    circuit_depth=4
)

db = QuantumDatabase(config)
```

## Complete Example: Document Search

```python
from q_store import QuantumDatabase, DatabaseConfig
import numpy as np

# Setup database (mock mode)
config = DatabaseConfig(
    quantum_sdk='mock',
    enable_quantum=True,
    enable_superposition=True,
    enable_tunneling=True
)

db = QuantumDatabase(config)

# Store documents with multiple contexts
documents = [
    {
        'id': 'doc_1',
        'vector': np.random.rand(768),
        'contexts': [('tech', 0.8), ('business', 0.2)]
    },
    {
        'id': 'doc_2',
        'vector': np.random.rand(768),
        'contexts': [('business', 0.7), ('legal', 0.3)]
    },
    {
        'id': 'doc_3',
        'vector': np.random.rand(768),
        'contexts': [('tech', 0.6), ('legal', 0.4)]
    }
]

# Insert all documents
for doc in documents:
    db.insert(
        id=doc['id'],
        vector=doc['vector'],
        contexts=doc['contexts']
    )

# Entangle related documents
db.create_entangled_group(
    group_id='related_docs',
    entity_ids=['doc_1', 'doc_3'],  # Both have tech context
    correlation_strength=0.85
)

# Query with context
query_vector = np.random.rand(768)

# Technical context query
tech_results = db.query(
    vector=query_vector,
    context='tech',
    top_k=5
)
print(f"Technical results: {[r.id for r in tech_results]}")

# Business context query
business_results = db.query(
    vector=query_vector,
    context='business',
    top_k=5
)
print(f"Business results: {[r.id for r in business_results]}")

# Quantum tunneling for diverse results
diverse_results = db.tunnel_search(
    query=query_vector,
    barrier_threshold=0.7,
    tunneling_strength=0.6,
    top_k=5
)
print(f"Diverse results: {[r.id for r in diverse_results]}")
```

## Performance Characteristics (v4.1.1)

**Real-World Data** (Cats vs Dogs, 1,000 images, 5 epochs):

| Backend | Training Time | Accuracy | Cost | Speedup | Use Case |
|---------|--------------|----------|------|---------|----------|
| **NVIDIA H100** | 5s | 60-70% | $0.009 | 457× faster | Production |
| **NVIDIA A100** | 7.5s | 60-70% | $0.010 | 305× faster | Production |
| **IonQ Simulator (v4.1)** | 38.1 min | 58.5% | $0 (free!) | Baseline | Research |
| **IonQ Aria QPU** | ~45-60 min | 60-75% | $1,152 | 0.15× | Research only |
| **Mock (v4.1)** | Instant | 10-20% | Free | N/A | Development |

**Key Insights**:
- ✅ v4.1.1 is **10-20× faster** than v4.0 (async execution)
- ⚠️ Classical GPUs are **183-457× faster** than quantum for large datasets
- ✅ **Free IonQ simulator** perfect for research and algorithm development
- ✅ Quantum excels: small datasets (<1K samples), non-convex optimization, research
- ⚠️ Use classical GPUs for: production, large datasets, time-critical applications

## Running Examples

### Available Examples (v4.1.1)

```bash
# Basic quantum circuit
python examples/basic_usage.py

# Async PyTorch model (Fashion MNIST)
python examples/pytorch/fashion_mnist_async.py --samples 500 --epochs 2

# Real-world benchmark (Cats vs Dogs)
python examples/pytorch/cats_vs_dogs.py --samples 1000 --epochs 5

# Database operations with async storage
python examples/database_async_demo.py

# Quantum-first architecture demo
python examples/quantum_first_demo.py

# Performance comparison (Quantum vs Classical)
python examples/performance_comparison.py
```

### With Real IonQ Simulator (Free!)

```bash
# Set up .env file with IONQ_API_KEY
python examples/pytorch/cats_vs_dogs.py --no-mock --samples 1000 --epochs 5

# Expected: ~38 minutes (vs 7.5s for GPU)
# Cost: $0 (simulator is free!)
# Perfect for research and algorithm development
```

## Tips for Success

1. **Start with Mock Mode**: Develop and test without API keys (instant, free)
2. **Use IonQ Simulator for Research**: Free, unlimited experimentation
3. **Understand Performance**: GPUs are 183-457× faster for production training
4. **Quantum Use Cases**: Research, small datasets (<1K), algorithm development, non-convex optimization
5. **Use Async Execution**: Enable async for 10-20× speedup over v4.0
6. **Optimal Qubit Counts**: 8 qubits recommended for v4.1.1 (was 4-8 in v4.0)
7. **Monitor Costs**: Real QPU costs $1,152-$4,480 per training run
8. **Network Latency**: 55% of execution time with cloud IonQ
9. **Energy Efficiency**: Quantum uses 5-8× less power than GPU (50-80W vs 400W)
10. **Check Examples**: All examples updated for v4.1.1 async architecture

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yucelz/q-store/issues)
- **Documentation**: [Q-Store Docs](/)
- **Examples**: `examples/` directory in repository
