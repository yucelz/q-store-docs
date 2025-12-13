---
title: IonQ Overview
description: Overview of IonQ quantum computing integration in Q-Store
---

Q-Store uses **IonQ** trapped-ion quantum computers as the quantum backend, providing high-fidelity quantum operations for database features.

## Why IonQ?

### Technical Advantages

**All-to-All Connectivity**
- Any qubit can interact with any other
- No SWAP gates needed
- Reduces circuit depth significantly

**High Fidelity**
- Single-qubit gates: >99.5% fidelity
- Two-qubit gates: >97% fidelity
- Industry-leading error rates

**Native Gate Set**
- Single-qubit: RX, RY, RZ, arbitrary rotations
- Two-qubit: XX gate (Mølmer-Sørensen)
- Efficient for our encoding schemes

**Scalability Roadmap**
- Current: 36 qubits (Forte)
- 2026: 64+ qubits
- 2028: 100+ qubits with error correction

## Available Systems

### Simulator (Free)

Perfect for development and testing.

```python
db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='simulator'
)
```

**Specifications:**
- Unlimited qubits (simulated)
- Free tier available
- Fast execution (<100ms)
- Perfect for CI/CD

### IonQ Aria (Production)

25-qubit system optimized for NISQ applications.

```python
db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='qpu.aria',
    ionq_api_key=YOUR_KEY
)
```

**Specifications:**
- 25 qubits
- #AQ 25 (Algorithmic Qubits)
- 99.5%+ single-qubit fidelity
- 97%+ two-qubit fidelity
- ~$0.30 per circuit

**Best For:**
- Production workloads
- Cost-sensitive applications
- Moderate complexity circuits

### IonQ Forte (Advanced)

36-qubit system for advanced quantum applications.

```python
db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='qpu.forte',
    ionq_api_key=YOUR_KEY
)
```

**Specifications:**
- 36 qubits
- #AQ 36 (Algorithmic Qubits)
- Highest fidelity available
- Advanced error mitigation
- ~$1.00 per circuit

**Best For:**
- Complex queries
- Large vector dimensions
- Research applications

### IonQ Forte Enterprise

Data center-ready quantum computing.

```python
db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='qpu.forte.1',
    ionq_api_key=YOUR_KEY
)
```

**Specifications:**
- 36 qubits
- Dedicated access
- SLA guarantees
- Enterprise support
- Custom pricing

**Best For:**
- Mission-critical applications
- High-volume production
- Enterprise deployments

## Getting Started with IonQ

### 1. Sign Up

1. Go to [cloud.ionq.com](https://cloud.ionq.com)
2. Create account (credit card required for QPU access)
3. Free simulator access immediately
4. QPU access upon approval

### 2. Generate API Key

```bash
# In IonQ Cloud Console
1. Navigate to "API Keys"
2. Click "Create New Key"
3. Copy and store securely
```

### 3. Configure Q-Store

```python
import os

# Set environment variable (recommended)
os.environ['IONQ_API_KEY'] = 'your-key-here'

# Or pass directly
db = QuantumDatabase(
    quantum_backend='ionq',
    ionq_api_key='your-key-here'
)
```

## Circuit Execution Flow

```
1. Q-Store builds circuit (Cirq)
   ↓
2. Optimize for IonQ native gates
   ↓
3. Submit to IonQ API
   ↓
4. IonQ compiles to hardware
   ↓
5. Execute on trapped ions
   ↓
6. Return measurement results
   ↓
7. Q-Store decodes results
```

## Performance Characteristics

### Latency

| Stage | Time | Description |
|-------|------|-------------|
| Circuit build | 10-50ms | Q-Store → Cirq circuit |
| Optimization | 20-100ms | Native gate decomposition |
| API submission | 10-50ms | HTTP to IonQ |
| Queue wait | 0-60s | Depends on load |
| Execution | 100-500ms | Quantum operations |
| Result retrieval | 10-50ms | Download results |
| **Total (simulator)** | **150-750ms** | End-to-end |
| **Total (QPU)** | **1-61s** | With queue time |

### Throughput

**Simulator:**
- Unlimited parallel jobs
- ~100 circuits/second per instance

**QPU (Aria/Forte):**
- Max 10 concurrent jobs
- ~5-10 circuits/second
- Shared access (queue varies)

## Cost Management

### Pricing (as of Dec 2025)

- **Simulator**: Free
- **Aria**: ~$0.30 per circuit
- **Forte**: ~$1.00 per circuit
- **Forte Enterprise**: Custom pricing

### Optimization Strategies

#### 1. Use Simulator for Development

```python
# Development
dev_db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='simulator'
)

# Production
prod_db = QuantumDatabase(
    quantum_backend='ionq',
    target_device='qpu.aria'
)
```

#### 2. Batch Operations

```python
# Expensive: Multiple submissions
for vector in vectors:
    db.query(vector)  # Multiple circuits

# Cheap: Single batched submission
db.query_batch(vectors)  # One combined circuit
```

#### 3. Circuit Caching

```python
# Enable caching
db = QuantumDatabase(
    quantum_backend='ionq',
    enable_cache=True,
    circuit_cache_size=1000
)
```

#### 4. Classical Pre-filtering

```python
# Good: Classical filter first
results = db.query(
    vector=query,
    classical_candidates=100,  # Filter to 100
    quantum_refine=True         # Then quantum
)

# Bad: All through quantum
results = db.query(
    vector=query,
    classical_candidates=10000,  # Too many!
    quantum_refine=True
)
```

## Error Handling

### Common Errors

**Authentication Error**
```python
try:
    db = QuantumDatabase(
        quantum_backend='ionq',
        ionq_api_key='invalid-key'
    )
except IonQAuthenticationError:
    print("Invalid API key")
```

**Quota Exceeded**
```python
try:
    results = db.query(vector)
except IonQQuotaExceeded:
    print("Monthly quota exceeded")
```

**Job Timeout**
```python
try:
    results = db.query(vector, timeout=120)
except IonQJobTimeout:
    print("Circuit execution timed out")
```

### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    retry_error_callback=lambda _: None
)
def execute_circuit(circuit):
    return ionq_backend.execute(circuit)
```

## Monitoring

### Job Status

```python
# Submit job
job = db.submit_circuit(circuit)

# Check status
status = job.get_status()
# 'queued', 'running', 'completed', 'failed'

# Wait for completion
results = job.wait(timeout=120)
```

### Metrics

```python
# Get usage statistics
stats = db.get_ionq_stats()

print(f"Circuits executed: {stats.circuits_executed}")
print(f"Total cost: ${stats.total_cost}")
print(f"Avg latency: {stats.avg_latency}ms")
print(f"Cache hit rate: {stats.cache_hit_rate}%")
```

## Next Steps

- Learn about [SDK Integration](/ionq/sdk-integration)
- Understand [Hardware Selection](/ionq/hardware-selection)
- See [IonQ Optimizations](/ionq/optimizations)
- Check [Production Deployment](/deployment/cloud)
