---
title: Old Versions (Retired)
description: Historical documentation for retired Q-Store versions 3.2, 3.3, 3.4, and 3.5
---

:::danger[All Versions on This Page Are Retired]
The versions documented below (v3.2, v3.3, v3.4, v3.5) are **retired and no longer supported**. Please upgrade to [v4.0](/getting-started/version-4-0) for the latest features, performance improvements, and production-ready optimizations.
:::

## Retired Versions Summary

This page contains historical documentation for retired Q-Store versions. These versions are no longer maintained or recommended for production use.

| Version | Release | Status | Key Features | Superseded By |
|---------|---------|--------|--------------|---------------|
| **v3.2** | Q4 2024 | ❌ Retired | Complete ML training capabilities, quantum neural networks | v3.5 |
| **v3.3** | Q4 2024 | ❌ Retired | 50-100x algorithmic optimization, SPSA gradient estimator | v3.5 |
| **v3.4** | Q4 2024 | ❌ Retired | 8-10x speed improvement, true parallelization | v3.5 |
| **v3.5** | Q1 2025 | ❌ Retired | 2-3x realistic improvements, multi-backend orchestration | v4.0 |

**Migration Recommendation**: All users should upgrade directly to [v4.0](/getting-started/version-4-0) for the best experience.

---

## Q-Store v3.2 (Retired)

:::caution[Retired Version]
This version is retired. Please upgrade to [v4.0](/getting-started/version-4-0) for better performance and production-ready optimizations.
:::

### Overview

Q-Store v3.2 introduced **complete machine learning training capabilities** with full hardware abstraction, enabling quantum neural networks that work seamlessly across simulators and quantum hardware.

### Key Features

#### Core ML Components
- **Quantum Neural Network Layers**: QuantumLayer, QuantumConvolutionalLayer, QuantumPoolingLayer
- **Gradient Computation**: Parameter Shift Rule, Finite Difference, Natural Gradients
- **Data Encoding**: Amplitude, Angle, Basis, and ZZ Feature Map encoding
- **Training Infrastructure**: Complete training orchestration with Adam optimizer, checkpoint management

#### Hardware Abstraction
Train once, run anywhere across different quantum backends (mock, Cirq, Qiskit, IonQ).

#### Transfer Learning
Pre-train and fine-tune models with parameter freezing support.

#### Performance Characteristics
- Qubits: 2-8 qubits
- Parameters: 6-48 trainable parameters
- Training Time: ~45 seconds for 5 epochs (mock backend)

### Basic Example

```python
from q_store.core import QuantumTrainer, QuantumModel, TrainingConfig, BackendManager

config = TrainingConfig(
    pinecone_api_key="your-api-key",
    quantum_sdk="mock",
    learning_rate=0.01,
    epochs=10,
    batch_size=5,
    n_qubits=4
)

backend_manager = BackendManager(config)
backend = backend_manager.get_backend("mock_ideal")

model = QuantumModel(
    input_dim=4,
    output_dim=2,
    n_layers=2,
    backend=backend
)

trainer = QuantumTrainer(config, backend_manager)
history = await trainer.train(model, data_loader)
```

---

## Q-Store v3.3 (Retired)

:::caution[Retired Version]
This version is retired. Please upgrade to [v4.0](/getting-started/version-4-0) for better performance and production-ready optimizations.
:::

### Overview

Q-Store v3.3 delivered **50-100x faster training** through algorithmic optimization while maintaining full backward compatibility with v3.2.

### Performance Improvements

| Metric | v3.2 | v3.3 | Improvement |
|--------|------|------|-------------|
| Circuits per batch | 960 | 10-20 | 48-96x |
| Time per batch | 240s | 5-10s | 24-48x |
| Time per epoch | 40min | 50-100s | 24-48x |
| Memory usage | 500MB | 200MB | 2.5x better |

### Core Components

#### 1. SPSA Gradient Estimator
Simultaneous Perturbation Stochastic Approximation - estimates ALL gradients with just 2 circuit evaluations instead of 2N evaluations for N parameters.

#### 2. Circuit Batch Manager
Batches multiple circuit executions into single API calls, reducing overhead by 5-10x.

#### 3. Intelligent Circuit Cache
Multi-level caching for quantum circuits to avoid redundant computations (2-5x speedup).

#### 4. Hardware-Efficient Ansatz
Optimized quantum layer with reduced gate count (33% fewer parameters).

#### 5. Adaptive Gradient Method Selector
Automatically selects the best gradient method based on training stage.

### Migration from v3.2

```python
# v3.2 code
config = TrainingConfig(
    learning_rate=0.01,
    batch_size=32,
    n_qubits=10,
    circuit_depth=4
)

# v3.3 - just add one line for 24x speedup!
config = TrainingConfig(
    learning_rate=0.01,
    batch_size=32,
    n_qubits=10,
    circuit_depth=4,
    gradient_method='spsa'  # Add this!
)
```

---

## Q-Store v3.4 (Retired)

:::caution[Retired Version]
This version is retired. Please upgrade to [v4.0](/getting-started/version-4-0) for better performance and production-ready optimizations.
:::

### Overview

Q-Store v3.4 delivered **8-10x faster training** through true parallelization and hardware-native optimizations, achieving sub-60 second training epochs on IonQ hardware.

### Performance Improvements

| Metric | v3.3 | v3.4 | Improvement |
|--------|------|------|-------------|
| Batch time (20 circuits) | 35s | 3-5s | 7-12x faster |
| Circuits per second | 0.6 | 5-8 | 8-13x faster |
| Epoch time | 392s | 30-50s | 8-13x faster |
| Training (5 epochs) | 32 min | 2.5-4 min | 8-13x faster |

### Core Components

#### 1. IonQBatchClient
True Parallel Batch Submission - submits multiple circuits in a single API call (12x faster submission).

#### 2. IonQNativeGateCompiler
Hardware-Native Gate Compilation - compiles to IonQ native gates (GPi, GPi2, MS) for 30% faster execution.

#### 3. SmartCircuitCache
Template-Based Circuit Caching - caches circuit structure and dynamically binds parameters (10x faster preparation).

#### 4. CircuitBatchManagerV34
Integrated Optimization Pipeline - orchestrates all v3.4 optimizations with comprehensive tracking.

#### 5. AdaptiveQueueManager
Dynamic Batch Optimization - adjusts batch size based on real-time queue conditions.

### Configuration

```python
config = TrainingConfig(
    # Enable all v3.4 features
    enable_all_v34_features=True,

    # Or selective enablement
    use_batch_api=True,
    use_native_gates=True,
    enable_smart_caching=True,
    connection_pool_size=5,
)
```

---

## Q-Store v3.5 (Retired)

:::caution[Retired Version]
This version is retired. Please upgrade to [v4.0](/getting-started/version-4-0) for better performance and production-ready optimizations.
:::

### Overview

Version 3.5 focused on **honest performance gains** and **addressing real bottlenecks** with realistic 2-3x performance improvements through proven techniques.

### Key Philosophy

- ✅ Realistic 2-3x improvement through proven techniques
- ✅ Multi-backend distribution for true parallel execution
- ✅ Adaptive resource allocation based on training phase
- ✅ Honest documentation with verified benchmarks

### What's New in v3.5

#### 1. Multi-Backend Orchestrator
Distribute quantum circuit execution across multiple backends simultaneously for 2-3x throughput improvement.

#### 2. Adaptive Circuit Optimization
Dynamically adjust circuit complexity during training for 30-40% faster execution.

#### 3. Adaptive Shot Allocation
Use minimum shots needed for gradient estimation, saving 20-30% execution time.

#### 4. Natural Gradient Descent
Replace SPSA with natural gradient for 2-3x fewer iterations to convergence.

### Verified Benchmarks

| Metric | v3.4 Actual | v3.5 Achieved |
|--------|-------------|---------------|
| Circuits/sec | 0.57 | 1.3 |
| Batch time (20 circuits) | 35s | 18s |
| Epoch time | 350s | 175s |
| Training (3 epochs) | 17.5 min | 8.5 min |
| Overall speedup | 1x | **2.1x** |

### Complete Configuration Example

```python
from q_store import TrainingConfig, QuantumTrainer

config = TrainingConfig(
    # Multi-backend orchestration
    enable_multi_backend=True,
    backend_configs=[
        {'provider': 'ionq', 'target': 'simulator', 'api_key': key1},
        {'provider': 'ionq', 'target': 'simulator', 'api_key': key2},
        {'provider': 'local', 'simulator': 'qiskit_aer', 'device': 'GPU'},
    ],

    # Adaptive optimizations
    adaptive_circuit_depth=True,
    circuit_depth_schedule='exponential',
    adaptive_shot_allocation=True,

    # Advanced gradient methods
    gradient_method='natural_gradient',

    # Enable all v3.5 features
    enable_all_v35_features=True,
)

trainer = QuantumTrainer(config)
await trainer.train(model, train_loader, val_loader)
```

### Known Limitations

- **Accuracy**: 70-75% on Fashion MNIST (vs 88-90% classical)
- **Inference speed**: ~2s per sample (vs <1ms classical)
- **Best for**: Parameter-limited scenarios, few-shot learning

---

## Migration Path

All users of retired versions should upgrade to **[v4.0](/getting-started/version-4-0)**:

1. **Update package**: `pip install --upgrade q-store>=4.0.0`
2. **Review v4.0 documentation**: Check the [v4.0 release notes](/getting-started/version-4-0)
3. **Update configuration**: Migrate to v4.0 configuration format
4. **Test thoroughly**: Validate performance and accuracy improvements

For migration assistance, please refer to the [v4.0 migration guide](/getting-started/version-4-0#migration-from-v3x) or open a [GitHub discussion](https://github.com/yucelz/q-store-docs/discussions).

---

**Last Updated**: December 2024
**Current Recommended Version**: [v4.0](/getting-started/version-4-0)
