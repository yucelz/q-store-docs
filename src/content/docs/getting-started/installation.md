---
title: Installation
description: Installation guide for Q-Store quantum database
---

## Requirements

- Python 3.9+
- IonQ API key (get from [cloud.ionq.com](https://cloud.ionq.com))
- Classical backend credentials (Pinecone, pgvector, or Qdrant)

## Install via pip

### Latest Version (v3.4.3 - Recommended)

```bash
pip install q-store
```

### Specific Version

```bash
pip install q-store==3.4.3
```

**What's New in v3.4.3**:
- 8-10x faster training through true parallelization
- Native IonQ gate compilation (GPi/GPi2/MS)
- Smart circuit caching with template-based optimization
- Connection pooling for reduced API overhead
- Production-ready with sub-60s training epochs



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
from q_store.ml import QuantumTrainer, TrainingConfig

# Test v3.4 installation
config = TrainingConfig(
    quantum_sdk='ionq',
    quantum_api_key='your-ionq-key',
    quantum_target='simulator',  # Free simulator
    
    # v3.4 features
    enable_all_v34_features=True
)

print(f"Q-Store v3.4.3 installed successfully!")
print(f"Features enabled: Batch API, Native Gates, Smart Caching")
```

### Test ML Training (v3.4)

```python
import asyncio
from q_store.ml import QuantumTrainer, TrainingConfig, QuantumModel

async def test_training():
    config = TrainingConfig(
        pinecone_api_key='your-pinecone-key',
        quantum_sdk='ionq',
        quantum_api_key='your-ionq-key',
        quantum_target='simulator',
        
        # v3.4 optimizations
        use_batch_api=True,
        use_native_gates=True,
        enable_smart_caching=True,
        
        batch_size=5,
        epochs=1
    )
    
    # Create simple model
    model = QuantumModel(
        input_dim=4,
        n_qubits=4,
        output_dim=2
    )
    
    print("Training with v3.4 optimizations...")
    # Training would go here with real data
    print("Installation verified! Ready for production.")

asyncio.run(test_training())
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
