---
title: Python SDK
description: Async Python SDK for Q-Store
---

The Python SDK provides both synchronous and asynchronous interfaces.

## Installation

```bash
pip install q-store
```

## Async Usage

```python
from quantum_db import QuantumDB, Config

# Initialize
config = Config(
    pinecone_api_key=PINECONE_KEY,
    ionq_api_key=IONQ_KEY
)

db = QuantumDB(config)

# Async operations
async with db.connect() as conn:
    # Insert
    await conn.insert(
        id="vec1",
        vector=embedding,
        contexts=[("ctx1", 0.7)]
    )
    
    # Query
    results = await conn.query(
        vector=query_embedding,
        context="ctx1",
        top_k=10
    )
```

## Sync Usage

```python
# Synchronous API
results = db.query_sync(
    vector=embedding,
    context="ctx1"
)
```

## Next Steps

See [Core API](/api/core) for detailed reference
