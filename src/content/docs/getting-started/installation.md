---
title: Installation
description: Installation guide for Q-Store quantum database
---

## Requirements

- Python 3.9+
- IonQ API key (get from [cloud.ionq.com](https://cloud.ionq.com))
- Classical backend credentials (Pinecone, pgvector, or Qdrant)

## Install via pip

```bash
pip install q-store
```

## Install from Source

```bash
git clone https://github.com/yucelz/q-store.git
cd q-store
pip install -e .
```

## Install with Optional Dependencies

### Pinecone Backend

```bash
pip install q-store[pinecone]
```

### pgvector Backend

```bash
pip install q-store[pgvector]
```

### Qdrant Backend

```bash
pip install q-store[qdrant]
```

### All Backends

```bash
pip install q-store[all]
```

## Setup IonQ Access

1. Sign up at [cloud.ionq.com](https://cloud.ionq.com)
2. Generate an API key
3. Set environment variable:

```bash
export IONQ_API_KEY='your-api-key-here'
```

Or configure in code:

```python
from quantum_db import QuantumDatabase

db = QuantumDatabase(
    quantum_backend='ionq',
    ionq_api_key='your-api-key-here'
)
```

## Setup Classical Backend

### Pinecone

```bash
export PINECONE_API_KEY='your-pinecone-key'
export PINECONE_ENVIRONMENT='us-east-1'
```

### pgvector

```python
db = QuantumDatabase(
    classical_backend='pgvector',
    classical_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'vectors',
        'user': 'postgres',
        'password': 'password'
    }
)
```

### Qdrant

```python
db = QuantumDatabase(
    classical_backend='qdrant',
    classical_config={
        'url': 'http://localhost:6333',
        'api_key': 'optional-api-key'
    }
)
```

## Verify Installation

```python
from quantum_db import QuantumDatabase

# Test basic functionality
db = QuantumDatabase(
    classical_backend='pinecone',
    quantum_backend='ionq',
    ionq_api_key=YOUR_KEY,
    target_device='simulator'  # Free simulator
)

# Insert test vector
db.insert(
    id='test',
    vector=[0.1, 0.2, 0.3, 0.4],
    contexts=[('test', 1.0)]
)

# Query
results = db.query(
    vector=[0.1, 0.2, 0.3, 0.4],
    top_k=1
)

print(f"Installation successful! Found {len(results)} results")
```

## Docker Installation

```bash
docker pull q-store/q-store:latest
docker run -e IONQ_API_KEY=your-key q-store/q-store:latest
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'cirq'**

```bash
pip install cirq cirq-ionq
```

**IonQ Authentication Error**

Verify your API key is correct and active at [cloud.ionq.com](https://cloud.ionq.com)

**Classical Backend Connection Error**

Check your backend credentials and network connectivity.

## Next Steps

- [Quick Start Guide](/getting-started/quick-start)
- [Architecture Overview](/concepts/architecture)
- [IonQ Integration](/ionq/overview)
