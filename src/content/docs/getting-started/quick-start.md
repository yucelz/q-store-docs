---
title: Quick Start
description: Get started with Q-Store v4.0.0 in 5 minutes
---

Get up and running with Q-Store v4.0.0 quantum-native database in minutes - no API keys required for development!

## Installation

```bash
# Latest version with v4.0.0 performance improvements
pip install q-store==4.0.0
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

## Hybrid Quantum-Classical ML

```python
import torch
import torch.nn as nn
from q_store.ml import QuantumLayer

# Build hybrid model
model = nn.Sequential(
    nn.Linear(784, 16),           # Classical layer
    QuantumLayer(n_qubits=4, depth=2),  # Quantum layer
    nn.Linear(4, 10)              # Classical output
)

# Train like any PyTorch model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    output = model(input_data)
    loss = criterion(output, labels)
    loss.backward()  # Quantum gradients computed automatically!
    optimizer.step()
```

**Run it:**
```bash
# Mock mode (instant, free)
python examples/pytorch/fashion_mnist.py --samples 500 --epochs 2

# Real quantum backend (requires API keys)
python examples/pytorch/fashion_mnist.py --no-mock --samples 100 --epochs 2
```

## Quantum-Enhanced Database

### Basic Setup

```python
from q_store import QuantumDatabase, DatabaseConfig

# Mock mode - no API keys needed
config = DatabaseConfig(
    quantum_sdk='mock',  # Use mock quantum backend
    enable_quantum=True,
    enable_superposition=True
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

## Performance Characteristics

| Mode | Accuracy | Speed | Cost | Use Case |
|------|----------|-------|------|----------|
| **Mock** | 10-20% | Instant | Free | Development, testing |
| **IonQ Simulator** | 60-75% | ~100ms | Low | Validation, demos |
| **IonQ QPU** | 60-75% | ~100ms | High | Production |

## Running Examples

### Available Examples

```bash
# Basic quantum circuit
python examples/basic_usage.py

# PyTorch hybrid model (Fashion MNIST)
python examples/pytorch/fashion_mnist.py --samples 500 --epochs 2

# Database operations
python examples/database_demo.py

# Financial applications
python examples/financial_demo.py

# Recommendation system
python examples/recommendation_demo.py
```

### With Real Quantum Backend

```bash
# Set up .env file first with API keys
python examples/pytorch/fashion_mnist.py --no-mock --samples 100 --epochs 2
```

## Tips for Success

1. **Start with Mock Mode**: Develop and test without API keys
2. **Use Small Qubit Counts**: 4-8 qubits optimal for v4.0.0
3. **Monitor Costs**: Real quantum operations cost money
4. **Validate Results**: Compare mock vs. real backend accuracy
5. **Check Examples**: All 11 examples pass validation tests

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yucelz/q-store/issues)
- **Documentation**: [Q-Store Docs](/)
- **Examples**: `examples/` directory in repository
