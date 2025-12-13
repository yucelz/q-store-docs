---
title: Classical Backend
description: Integration with classical vector databases
---

The **Classical Backend** handles bulk storage and coarse filtering using mature vector databases.

## Supported Backends

- **Pinecone**: Managed vector database
- **pgvector**: PostgreSQL extension
- **Qdrant**: Open-source vector database
- **Redis**: Caching layer

## Configuration

```python
# Pinecone
db = QuantumDatabase(
    classical_backend='pinecone',
    classical_config={
        'api_key': PINECONE_KEY,
        'environment': 'us-east-1'
    }
)

# pgvector
db = QuantumDatabase(
    classical_backend='pgvector',
    classical_config={
        'host': 'localhost',
        'database': 'vectors'
    }
)
```

## Next Steps

See other [System Components](/components/state-manager)
