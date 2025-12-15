---
title: Hybrid Design Rationale
description: Why Q-Store uses a hybrid classical-quantum architecture
---

## The NISQ Reality

We're currently in the **NISQ era** (Noisy Intermediate-Scale Quantum):

### Current Limitations
- **Limited qubits**: 10-100 available (IonQ Forte: 36 qubits)
- **Short coherence times**: Milliseconds to seconds
- **Gate errors**: 1-5% error rates
- **High cost**: ~$0.30-$1.00 per circuit execution
- **No error correction**: Must work with noise

### Why Not Pure Quantum?

A pure quantum database would:
- Store only ~100-1,000 vectors (qubit limitations)
- Require frequent re-encoding (coherence limits)
- Cost prohibitively for large datasets
- Lack maturity of classical systems

## Hybrid Advantages

The hybrid approach gives us **best of both worlds**:

### Classical Strengths
✓ Proven reliability  
✓ Mature ecosystems  
✓ Cost-effective bulk storage  
✓ Billions of vectors  
✓ Established tooling  

### Quantum Strengths
✓ Exponential speedup (O(√N))  
✓ Context-aware superposition  
✓ Entangled relationships  
✓ Pattern tunneling  
✓ Novel capabilities  

## Division of Labor

### Classical Component Handles
- **Bulk storage**: Millions/billions of vectors
- **Cold data**: Rarely accessed items
- **Coarse filtering**: Top-1000 candidates
- **Persistent storage**: Long-term retention
- **Backup/failover**: When quantum unavailable

### Quantum Component Handles
- **Hot data**: 100-1,000 active vectors
- **Context resolution**: Superposition collapse
- **Relationship management**: Entanglement
- **Pattern discovery**: Quantum tunneling
- **Fine-grained ranking**: Final top-K

## Query Pipeline

### Multi-Stage Approach

```
1. Parse Query
   ↓
2. Classical Filter (Pinecone/pgvector)
   • Filter 99% of data
   • Return top-1000 candidates
   • Fast, cheap operation
   ↓
3. Quantum Refinement (IonQ)
   • Encode 100-1000 candidates
   • Apply superposition contexts
   • Enable tunneling search
   • Quantum measurement
   ↓
4. Post-processing
   • Ranking
   • Deduplication
   • Result formatting
```

### Why This Works

**Classical pre-filter reduces quantum load by 99%+**

Example with 1M vectors:
- Classical filter: 1M → 1,000 (0.1% retained)
- Quantum refinement: 1,000 → 10 (final results)
- Quantum processes <0.1% of data
- Gets 90%+ of accuracy benefit

## Performance Trade-offs

### Latency Breakdown

| Stage | Time | Backend |
|-------|------|---------|
| Parse | 1ms | Application |
| Classical filter | 10-50ms | Pinecone/pgvector |
| Quantum encode | 50-100ms | Circuit preparation |
| Quantum execute | 100-500ms | IonQ hardware |
| Post-process | 5-10ms | Application |
| **Total** | **166-661ms** | **End-to-end** |

### When to Use Quantum

**Good fit:**
- Context matters (superposition valuable)
- Relationships complex (entanglement helps)
- Patterns hidden (tunneling needed)
- Quality > latency

**Poor fit:**
- Simple nearest-neighbor only
- Ultra-low latency required (<50ms)
- No context/relationships
- Cost-sensitive high-volume

## Cost Optimization

### Minimize Quantum Calls

```python
# Good: Aggressive classical filtering
results = db.query(
    vector=query,
    classical_candidates=1000,  # Filter first
    quantum_refine=True,        # Then quantum
    top_k=10
)

# Bad: Everything through quantum
results = db.query(
    vector=query,
    classical_candidates=100000,  # Too many!
    quantum_refine=True,
    top_k=10
)
```

### Use Simulator for Development

```python
# Free simulator for development
dev_db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='simulator'  # Free!
)

# Real hardware for production
prod_db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='qpu.aria'  # $0.30/circuit
)
```

### Cache Aggressively

```python
# Circuit caching (v2.0)
db = QuantumDatabase(
    enable_cache=True,
    cache_ttl=300,  # 5 minutes
    circuit_cache_size=1000
)
```


### Migration Path

```
Phase 1 (Now): Hybrid with 10% quantum enhancement
Phase 2 (2027): Hybrid with 50% quantum processing  
Phase 3 (2030): Quantum-primary with classical backup
Phase 4 (2035+): Pure quantum (post-NISQ era)
```

## Classical Backend Options

### Pinecone
- **Pros**: Managed service, easy setup, good performance
- **Cons**: Costs scale with usage
- **Best for**: Quick start, production workloads

### pgvector
- **Pros**: Open source, PostgreSQL integration, flexible
- **Cons**: Requires self-hosting
- **Best for**: Existing PostgreSQL users, custom setups

### Qdrant
- **Pros**: Open source, high performance, rich features
- **Cons**: Newer ecosystem
- **Best for**: Advanced users, specific requirements

### Redis (v2.0 caching)
- **Pros**: Ultra-fast, widely used
- **Cons**: In-memory only
- **Best for**: L1/L2 cache layer

## Next Steps

- Understand [Quantum Principles](/concepts/quantum-principles)
- Learn about [System Components](/components/state-manager)
- See [IonQ Integration](/ionq/overview) for quantum backend
- Check [Production Patterns](/production/connection-pooling) for deployment
