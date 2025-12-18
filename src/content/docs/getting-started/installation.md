---
title: Installation
description: Installation guide for Q-Store quantum database
---

## Requirements

- Python 3.9+
- IonQ API key (get from [cloud.ionq.com](https://cloud.ionq.com))
- Classical backend credentials (Pinecone, pgvector, or Qdrant)
- Optional: GPU for local quantum simulation acceleration

## Install via pip

### Latest Version (v3.5.0 - Recommended)

```bash
pip install q-store
```

### Specific Version

```bash
pip install q-store==3.5.0
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
- [v3.4 Release Notes](/getting-started/version-3-4)
- [Architecture Overview](/concepts/architecture)
- [IonQ Integration](/ionq/overview)
