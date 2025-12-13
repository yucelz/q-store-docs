---
title: Quantum Principles
description: Core quantum mechanical principles exploited by Q-Store
---

Q-Store leverages four fundamental quantum properties as **features** rather than obstacles:

## 1. Superposition as Multi-Version Storage

### Classical Problem
A vector has one fixed value at a time. Supporting multiple contexts requires:
- Storing duplicate copies
- Complex branching logic
- Manual context management

### Quantum Solution
Store vectors in **superposition** of multiple states simultaneously.

```python
# Classical: One definite vector
vector = [0.5, 0.3, 0.8]

# Quantum: Superposition of multiple interpretations
|ψ⟩ = α|context_A⟩ + β|context_B⟩ + γ|context_C⟩
    = 0.7|"normal_usage"⟩ + 0.2|"edge_case"⟩ + 0.1|"legacy"⟩
```

### When Queried
Measurement **collapses** to the version most relevant to query context.

### Advantages
- One quantum state = multiple classical versions
- Context-aware without explicit branching logic
- **Exponential compression**: n qubits = 2^n states

### Example Usage

```python
# Insert with multiple contexts
db.insert(
    id='doc_123',
    vector=embedding,
    contexts=[
        ('technical', 0.6),    # 60% probability
        ('general', 0.3),      # 30% probability
        ('historical', 0.1)    # 10% probability
    ]
)

# Query collapses to relevant context
results = db.query(
    vector=query_embedding,
    context='technical'  # Superposition collapses here
)
```

## 2. Entanglement for Relational Integrity

### Classical Problem
- Related entities stored separately
- Consistency requires explicit synchronization
- Cache invalidation complexity
- Race conditions possible

### Quantum Solution
**Entangle** related entities; updates propagate automatically via quantum correlation.

```python
# Entangled state for correlated data
|Ψ⟩ = 1/√2(|A₁⟩|B₁⟩ + |A₂⟩|B₂⟩)

# Update A → B automatically updates (quantum non-locality)
# Physically impossible for A and B to desync
```

### Advantages
- **Zero-latency** relationship updates
- Impossible to have stale references
- No cache invalidation logic needed
- Correlation strength = entanglement entropy

### Example Usage

```python
# Create entangled group
db.create_entangled_group(
    group_id='related_stocks',
    entity_ids=['AAPL', 'MSFT', 'GOOGL'],
    correlation_strength=0.85
)

# Update one entity
db.update('AAPL', new_embedding)

# MSFT and GOOGL automatically reflect correlation
# No manual sync required!
```

## 3. Decoherence as Adaptive TTL

### Classical Problem
- Manual cache expiry policies
- Complex TTL management
- No natural relevance decay
- Over-retention or premature deletion

### Quantum Solution
**Coherence time** = relevance; physics handles expiry automatically.

```python
# Different data types get different coherence times
hot_data:    coherence_time = 1000ms  # Stays relevant
normal_data: coherence_time = 100ms   # Fades naturally
critical:    coherence_time = 10000ms # Always remembered
```

### Advantages
- No explicit TTL management
- Physics-based relevance decay
- Adaptive memory without code
- Reduces storage costs automatically

### Example Usage

```python
# Hot data - long coherence
db.insert(
    id='trending_topic',
    vector=embedding,
    coherence_time=5000  # 5 seconds
)

# Normal data - medium coherence
db.insert(
    id='regular_doc',
    vector=embedding,
    coherence_time=1000  # 1 second
)

# Cleanup happens naturally
db.apply_decoherence()  # Old data fades away
```

## 4. Quantum Tunneling for Pattern Discovery

### Classical Problem
- Local optima traps in search
- Can't find globally optimal patterns
- Misses rare but important signals
- Stuck in nearest-neighbor regions

### Quantum Solution
**Quantum tunneling** passes through barriers to reach distant patterns.

```python
# Classical: A → nearby B (local optimum)
# Quantum:  A → tunnel through barrier → distant C (global optimum)
```

### Enables Discovery Of
- Pre-crisis patterns that look "normal" classically
- Semantic matches with different syntax
- Hidden correlations in high-dimensional space
- Unexpected but valuable connections

### Advantages
- Finds patterns classical ML misses
- Escapes local optima in training
- **O(√N) vs O(N)** search complexity

### Example Usage

```python
# Enable tunneling for pattern discovery
results = db.query(
    vector=query_embedding,
    enable_tunneling=True,
    barrier_threshold=0.8,  # How far to tunnel
    top_k=10
)

# Returns distant but relevant matches
# Classical search would miss these
```

## 5. Uncertainty Principle for Explicit Tradeoffs

### Classical Problem
- Hidden tradeoffs between precision and coverage
- No way to control explicitly
- Opaque search behavior

### Quantum Solution
Heisenberg uncertainty makes tradeoff **explicit and optimal**.

```
ΔPrecision · ΔCoverage ≥ ℏ/2
```

### User Control

```python
# Precise mode: High precision, lower coverage
results = db.query(mode='precise')

# Exploratory mode: Lower precision, high coverage  
results = db.query(mode='exploratory')

# Balanced: Quantum-optimal tradeoff
results = db.query(mode='balanced')
```

### Advantages
- Explicit control over tradeoffs
- Quantum-optimal balance
- No hidden compromises
- Transparent behavior

## Mathematical Foundation

### Amplitude Encoding

```
|ψ⟩ = Σᵢ αᵢ|i⟩

where:
- αᵢ = normalized vector components
- |αᵢ|² = probability of measuring state |i⟩
- Σᵢ |αᵢ|² = 1 (normalization)
```

### Entanglement Measure

```
S(ρ) = -Tr(ρ log ρ)

where:
- S(ρ) = von Neumann entropy
- ρ = reduced density matrix
- Higher S = stronger entanglement
```

### Tunneling Probability

```
T ≈ exp(-2κL)

where:
- T = transmission coefficient
- κ = barrier strength
- L = barrier width
```

## Next Steps

- See [Hybrid Design](/concepts/hybrid-design) for architecture
- Explore [IonQ Integration](/ionq/overview) for implementation
- Check [Domain Applications](/applications/financial) for use cases
