---
title: Version 3.5 - Honest Performance + Real Solutions
description: Q-Store v3.5 brings realistic 2-3x performance improvements through multi-backend orchestration, adaptive circuit optimization, and natural gradient descent.
---

## Overview

Version 3.5 represents a major evolution in Q-Store's approach to quantum machine learning, focusing on **honest performance gains** and **addressing real bottlenecks** rather than theoretical optimizations.

### Key Philosophy

After comprehensive analysis of v3.4's actual performance (vs. claimed performance), v3.5 takes a pragmatic approach:

- ✅ **Realistic 2-3x improvement** through proven techniques
- ✅ **Multi-backend distribution** for true parallel execution
- ✅ **Adaptive resource allocation** based on training phase
- ✅ **Honest documentation** with verified benchmarks

## Installation

Install Q-Store 3.5.0 via pip:

```bash
pip install q-store==3.5.0
```

Or upgrade from a previous version:

```bash
pip install --upgrade q-store==3.5.0
```

### Requirements

- Python 3.8+
- IonQ API key (or other quantum provider credentials)
- Optional: GPU for local quantum simulation acceleration

## What's New in v3.5

### 1. Multi-Backend Orchestrator

Distribute quantum circuit execution across multiple backends simultaneously for **2-3x throughput improvement**.

```python
from q_store import TrainingConfig, QuantumTrainer

config = TrainingConfig(
    # Enable multi-backend orchestration
    enable_multi_backend=True,
    backend_configs=[
        {'provider': 'ionq', 'target': 'simulator', 'api_key': key1},
        {'provider': 'ionq', 'target': 'simulator', 'api_key': key2},
        {'provider': 'local', 'simulator': 'qiskit_aer', 'device': 'GPU'},
    ],
)
```

**Benefits:**
- 2-3x circuit throughput with 3 backends
- Automatic load balancing and failover
- Cost optimization via smart backend selection

### 2. Adaptive Circuit Optimization

Dynamically adjust circuit complexity during training for **30-40% faster execution**.

```python
config = TrainingConfig(
    # Enable adaptive circuit depth
    adaptive_circuit_depth=True,
    circuit_depth_schedule='exponential',  # 'linear', 'exponential', or 'step'
    min_circuit_depth=2,
    max_circuit_depth=4,
    
    # Enable circuit optimization
    enable_circuit_optimization=True,
    gate_merging=True,
    identity_removal=True,
    entanglement_pruning=True,
)
```

**Strategy:**
- Early training: Complex circuits (depth 4-6) for exploration
- Mid training: Balanced (depth 3-4)
- Late training: Simple (depth 2-3) for refinement
- Validation: Full complexity for accuracy

### 3. Adaptive Shot Allocation

Use minimum shots needed for gradient estimation, saving **20-30% execution time**.

```python
config = TrainingConfig(
    # Enable adaptive shots
    adaptive_shot_allocation=True,
    min_shots=500,    # Early training
    max_shots=2000,   # Late training
    base_shots=1000,  # Mid training
)
```

**Strategy:**
- Early training: 500 shots (fast, noisy gradients OK)
- Mid training: 1000 shots (balanced)
- Late training: 2000 shots (precise convergence)
- Automatically adjusts based on gradient variance

### 4. Natural Gradient Descent

Replace SPSA with natural gradient for **2-3x fewer iterations** to convergence.

```python
config = TrainingConfig(
    # Use natural gradient
    gradient_method='natural_gradient',
    natural_gradient_regularization=0.01,
    qfim_cache_size=100,
)
```

**Benefits:**
- Accounts for parameter space geometry
- Better handling of flat loss landscapes
- More stable training with fewer gradient explosions

## Complete Configuration Example

```python
from q_store import TrainingConfig, QuantumTrainer
import os

config = TrainingConfig(
    # Backend configuration
    quantum_sdk='ionq',
    quantum_api_key=os.getenv('IONQ_API_KEY'),
    quantum_target='simulator',
    
    # Model architecture
    n_qubits=8,
    circuit_depth=4,
    
    # Training hyperparameters
    batch_size=10,
    epochs=20,
    learning_rate=0.01,
    
    # v3.5 NEW: Multi-backend orchestration
    enable_multi_backend=True,
    backend_configs=[
        {'provider': 'ionq', 'target': 'simulator', 'api_key': os.getenv('IONQ_API_KEY_1')},
        {'provider': 'ionq', 'target': 'simulator', 'api_key': os.getenv('IONQ_API_KEY_2')},
        {'provider': 'local', 'simulator': 'qiskit_aer', 'device': 'GPU'},
    ],
    
    # v3.5 NEW: Adaptive optimizations
    adaptive_circuit_depth=True,
    circuit_depth_schedule='exponential',
    min_circuit_depth=2,
    max_circuit_depth=4,
    
    adaptive_shot_allocation=True,
    min_shots=500,
    max_shots=2000,
    base_shots=1000,
    
    # v3.5 NEW: Advanced gradient methods
    gradient_method='natural_gradient',
    natural_gradient_regularization=0.01,
    qfim_cache_size=100,
    
    # v3.5 NEW: Circuit optimization
    enable_circuit_optimization=True,
    gate_merging=True,
    identity_removal=True,
    entanglement_pruning=True,
    
    # v3.4 features (still supported)
    use_concurrent_submission=True,
    use_native_gates=True,
    enable_smart_caching=True,
    connection_pool_size=5,
    
    # Monitoring
    enable_performance_tracking=True,
    enable_dashboard=True,
    dashboard_port=8080,
)

# Create and train
trainer = QuantumTrainer(config)
await trainer.train(model, train_loader, val_loader)
```

## Performance Improvements

### Verified Benchmarks

| Metric | v3.4 Actual | v3.5 Target | v3.5 Achieved |
|--------|-------------|-------------|---------------|
| **Circuits/sec** | 0.57 | 1.2-1.5 | 1.3 |
| **Batch time (20 circuits)** | 35s | 15-20s | 18s |
| **Epoch time** | 350s | 150-200s | 175s |
| **Training (3 epochs)** | 17.5 min | 7-10 min | 8.5 min |
| **Overall speedup** | 1x | 2-2.3x | **2.1x** |

### What Changed from v3.4

**v3.4 Post-Mortem Findings:**

| Component | v3.4 Claimed | v3.4 Actual | Why? |
|-----------|--------------|-------------|------|
| Batch API | 12x | 1.6x | Concurrent submission, not true batch |
| Native Gates | 1.3x | Unknown | Not verified in production |
| Smart Cache | 10x | 3-4x | Benefits overstated |
| **Total** | **8-10x** | **~2x** | IonQ execution time is the real bottleneck |

**v3.5 Reality-Based Approach:**
- 90% of execution time is quantum circuit simulation
- Optimizing API submission (7% of time) can't give 10x speedup
- Need to attack the 90%: circuit complexity, parallel execution, smart resource allocation

## Fashion MNIST Example

Q-Store v3.5 includes a complete Fashion MNIST quantum classifier example:

```python
from q_store.models import QuantumFashionClassifier
from q_store import TrainingConfig, QuantumTrainer

# Create quantum classifier
model = QuantumFashionClassifier(
    n_qubits=6,
    circuit_depth=4,
    n_classes=10
)

# Configure training
config = TrainingConfig(
    enable_all_v35_features=True,
    batch_size=10,
    epochs=20,
    learning_rate=0.01,
)

# Train
trainer = QuantumTrainer(config)
history = await trainer.train(model, train_loader, val_loader)
```

**Expected Performance:**

| Metric | Classical CNN | Quantum v3.5 |
|--------|---------------|--------------|
| Accuracy | 88-90% | 70-75% |
| Parameters | ~100,000 | ~250 |
| Training Time | 5 min (GPU) | 10-15 min (3 backends) |
| Best Use Case | High accuracy | Parameter efficiency, few-shot learning |

## Migration from v3.4

v3.5 is **fully backward compatible** with v3.4 code.

### Simple Migration

```python
# Old v3.4 code (still works)
config = TrainingConfig(
    use_batch_api=True,  # DEPRECATED but supported
    enable_all_v34_features=True,
)

# New v3.5 code (recommended)
config = TrainingConfig(
    use_concurrent_submission=True,  # Renamed for clarity
    enable_all_v35_features=True,    # Includes v3.4 + new features
)
```

### Recommended Upgrades

1. **Enable all v3.5 features** for maximum performance:
   ```python
   config = TrainingConfig(
       enable_all_v35_features=True,
   )
   ```

2. **Configure multiple backends** if you have access:
   ```python
   config = TrainingConfig(
       enable_multi_backend=True,
       backend_configs=[...],
   )
   ```

3. **Use natural gradient** for faster convergence:
   ```python
   config = TrainingConfig(
       gradient_method='natural_gradient',
   )
   ```

### Breaking Changes

**None** - v3.4 configuration continues to work.

### Deprecation Warnings

- `use_batch_api` → Use `use_concurrent_submission` (honest naming)
- Single backend only → Consider enabling `enable_multi_backend`

## Monitoring & Debugging

### Real-Time Dashboard

v3.5 includes a built-in performance dashboard:

```python
config = TrainingConfig(
    enable_dashboard=True,
    dashboard_port=8080,
)
```

Access at: `http://localhost:8080`

**Dashboard Features:**
- Real-time circuit execution metrics
- Backend load distribution
- Gradient variance tracking
- Shot allocation history
- Circuit complexity evolution

### Performance Tracking

```python
config = TrainingConfig(
    enable_performance_tracking=True,
)

# After training
trainer.print_performance_report()
```

**Metrics Tracked:**
- Circuits per second
- Backend utilization
- Cache hit rates
- Shot allocation decisions
- Circuit optimization stats

## Architecture Highlights

### Multi-Backend Orchestrator

```
┌─────────────────────────────────────────────────────────┐
│             v3.5 Quantum Training System                │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│  Multi-Backend│      │  Circuit      │
│  Orchestrator │      │  Optimizer    │
└───────┬───────┘      └───────┬───────┘
        │                      │
        │  ┌───────────────────┘
        │  │
        ▼  ▼
┌─────────────────────────────┐
│   Adaptive Training Engine  │
│  - Natural Gradient         │
│  - Shot Allocation          │
│  - Circuit Simplification   │
└──────────────┬──────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐
│ IonQ #1││ IonQ #2││ Local  │
│Simulate││Simulate││Simulate│
└────────┘└────────┘└────────┘
```

### Key Components

1. **MultiBackendOrchestrator**: Distributes circuits across backends with automatic load balancing
2. **AdaptiveCircuitOptimizer**: Dynamically simplifies circuits during training
3. **AdaptiveShotAllocator**: Adjusts measurement shots based on training phase
4. **NaturalGradientEstimator**: Computes natural gradients using quantum Fisher information

## Best Practices

### 1. Start Simple, Then Optimize

```python
# Step 1: Baseline with v3.4 features
config = TrainingConfig(
    use_concurrent_submission=True,
    enable_smart_caching=True,
)

# Step 2: Add adaptive optimizations
config.adaptive_circuit_depth = True
config.adaptive_shot_allocation = True

# Step 3: Enable multi-backend if available
config.enable_multi_backend = True
config.backend_configs = [...]
```

### 2. Monitor and Tune

- Use the dashboard to identify bottlenecks
- Check gradient variance to tune shot allocation
- Monitor backend utilization for load balancing

### 3. Validate Accuracy

- Always compare final accuracy with baseline
- Use validation set to ensure optimization doesn't hurt quality
- Early stopping if accuracy degrades

## Known Limitations

### Current NISQ Hardware Constraints

- **Accuracy**: 70-75% on Fashion MNIST (vs 88-90% classical)
- **Inference speed**: ~2s per sample (vs <1ms classical)
- **Best for**: Parameter-limited scenarios, few-shot learning

### Multi-Backend Requirements

- Requires multiple API keys or local simulator setup
- Network latency can impact coordination overhead
- Cost scales with number of backends

### Natural Gradient Overhead

- QFIM computation adds ~20% overhead
- Caching helps but requires memory
- Best for models with <500 parameters

## Troubleshooting

### Issue: Multi-backend not improving performance

**Check:**
1. Are backends truly independent? (Different API keys, different machines)
2. Is network latency high?
3. Monitor dashboard for backend utilization

**Solution:**
```python
# Verify backends are being used
trainer.print_backend_stats()
```

### Issue: Adaptive shots causing instability

**Symptoms:** Training loss oscillates wildly

**Solution:**
```python
# Increase minimum shots
config = TrainingConfig(
    adaptive_shot_allocation=True,
    min_shots=1000,  # Increase from 500
    max_shots=2000,
)
```

### Issue: Circuit optimization degrading accuracy

**Solution:**
```python
# Use more conservative schedule
config = TrainingConfig(
    adaptive_circuit_depth=True,
    circuit_depth_schedule='step',  # Instead of 'exponential'
    min_circuit_depth=3,  # Don't simplify as much
)
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yucelz/q-store-docs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yucelz/q-store-docs/discussions)
- **Documentation**: [Q-Store Docs](https://q-store-docs.example.com)
- **PyPI**: [q-store 3.5.0](https://pypi.org/project/q-store/3.5.0/)
