---
title: Cost Optimization
description: Strategies for optimizing Q-Store costs
---

Cost optimization strategies for production Q-Store deployments.

## Cost Model

### Classical Costs
- Pinecone: $0.096/hour per pod
- Redis Cache: $0.05/GB-month
- Compute: $0.10/hour per instance

### Quantum Costs
- IonQ Simulator: Free
- IonQ Aria: ~$0.30 per circuit
- IonQ Forte: ~$1.00 per circuit

## Optimization Strategies

### 1. Aggressive Classical Filtering

```python
# Reduce quantum load by 90%+
results = db.query(
    vector=query,
    classical_candidates=100,  # Filter first
    quantum_refine=True
)
```

### 2. Circuit Caching

```python
db = QuantumDatabase(
    enable_cache=True,
    circuit_cache_size=1000
)
```

### 3. Batch Operations

```python
# Amortize quantum overhead
db.insert_batch(vectors)
```

### 4. Use Simulator for Development

```python
dev_db = QuantumDatabase(
    target_device='simulator'  # Free
)
```

### 5. Smart Routing

Only use quantum when beneficial:

```python
if context_matters:
    results = db.query(quantum_refine=True)
else:
    results = db.query(quantum_refine=False)
```

## Cost Example

**Portfolio: 1000 stocks, 10K queries/day**

- Classical: ~$100/month
- + Quantum (10%): ~$300/month
- **Total: ~$400/month**

## Next Steps

See [Performance](/advanced/performance) and [Testing](/advanced/testing)
