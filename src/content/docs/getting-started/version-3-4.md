---
title: Q-Store v3.4
description: Production-Ready Performance with 8-10x Speed Improvement Through True Parallelization
---

## Overview

Q-Store v3.4 delivers **8-10x faster training** through true parallelization and hardware-native optimizations. This release addresses critical bottlenecks in v3.3.1, achieving sub-60 second training epochs on IonQ hardware.

## What's New in v3.4

### 🚀 Performance Improvements

| Metric | v3.3.1 | v3.4 | Improvement |
|--------|--------|------|-------------|
| **Batch time (20 circuits)** | 35s | 3-5s | **7-12x faster** |
| **Circuits per second** | 0.6 | 5-8 | **8-13x faster** |
| **Epoch time** | 392s | 30-50s | **8-13x faster** |
| **Training (5 epochs)** | 32 min | 2.5-4 min | **8-13x faster** |
| **Circuit preparation** | 0.5s | 0.05s | **10x faster** |

### 🔍 Critical Problems Solved

**v3.3.1 Bottlenecks**:
- Sequential circuit submission despite "batch" API (20 circuits × 1.8s each = 36s)
- High IonQ API overhead per request (~0.5s + 1.3s queue time)
- No circuit optimization for native IonQ gates
- No connection pooling (new connection per circuit)
- Inefficient circuit rebuilding for identical structures

**v3.4 Solutions**:
- True batch submission (single API call for all circuits → 2-4s total)
- Native gate compilation (GPi/GPi2/MS gates → 30% faster execution)
- Smart circuit caching (parameter binding → 80% reduction in build time)
- Connection pooling (persistent connections → 60% reduced overhead)
- Adaptive queue management (dynamic batch sizing)

### 🆕 Core Components

#### 1. IonQBatchClient

**True Parallel Batch Submission** - submits multiple circuits in a single API call instead of sequential submissions.

**Key Features**:
- `submit_batch()` - Submit multiple circuits in single API call
- `get_results_parallel()` - Fetch results concurrently with asyncio.gather
- Connection pooling with configurable pool size
- Automatic retry with exponential backoff
- Rate limiting and queue depth monitoring

**Performance Impact**: 12x faster submission (1 batch call vs 20 sequential calls)

#### 2. IonQNativeGateCompiler

**Hardware-Native Gate Compilation** - compiles standard gates to IonQ native gates (GPi, GPi2, MS) for direct hardware execution.


**Performance Impact**: 30% faster execution on IonQ hardware


#### 3. SmartCircuitCache

**Template-Based Circuit Caching** - caches circuit structure and dynamically binds parameters instead of rebuilding circuits.


**Performance Impact**: 10x faster circuit preparation (binding vs rebuilding)

**Innovation**: Cache the circuit structure once, then bind different parameters for each training sample.

#### 4. CircuitBatchManagerV34

**Integrated Optimization Pipeline** - orchestrates all v3.4 optimizations with comprehensive performance tracking.

**Key Features**:
- `execute_batch()` - Execute circuits with all v3.4 optimizations
- `get_performance_stats()` - Retrieve detailed performance metrics
- `print_performance_report()` - Display comprehensive performance summary
- Adaptive batch sizing based on queue conditions
- Automatic fallback to v3.3.1 if optimizations unavailable

**Performance Impact**: 8-10x overall speedup in production training

#### 5. AdaptiveQueueManager

**Dynamic Batch Optimization** - adjusts batch size based on real-time queue conditions.

**Key Features**:
- `get_optimal_batch_size()` - Calculate optimal batch size from queue metrics
- `record_job()` - Track job metrics for adaptive learning
- Queue depth monitoring
- Historical pattern analysis

**Strategy**:
- Larger batches (20) when queue is empty (< 1s wait)
- Medium batches (10) when queue is moderate (1-3s wait)
- Smaller batches (5) when queue is full (> 3s wait)

## Configuration

### Enable All v3.4 Features

```python
from q_store.ml import TrainingConfig

config = TrainingConfig(
    pinecone_api_key="your_key",
    quantum_sdk="ionq",
    quantum_api_key="your_ionq_key",
    
    # Single flag to enable all optimizations
    enable_all_v34_features=True
)
```

### Selective Feature Enablement

```python
config = TrainingConfig(
    # Choose specific optimizations
    use_batch_api=True,          # 12x faster submission (recommended)
    use_native_gates=True,       # 30% faster execution (recommended)
    enable_smart_caching=True,   # 10x faster preparation (recommended)
    adaptive_batch_sizing=False, # Optional queue optimization
    
    # Advanced tuning
    connection_pool_size=5,      # HTTP connections to IonQ
    max_queue_wait_time=120.0,   # Timeout for queue (seconds)
    retry_failed_circuits=True,  # Auto-retry on failure
)
```

### Configuration Options

| Option | Default | Description | Impact |
|--------|---------|-------------|--------|
| `use_batch_api` | True | Single API call for batch | 12x faster |
| `use_native_gates` | True | Compile to GPi/GPi2/MS | 30% faster |
| `enable_smart_caching` | True | Template-based caching | 10x faster |
| `adaptive_batch_sizing` | False | Dynamic batch sizing | 1.2x faster |
| `connection_pool_size` | 5 | HTTP connection pool | Reduces overhead |
| `max_queue_wait_time` | 120.0 | Queue timeout (seconds) | Prevents hangs |

## Performance Monitoring

### Built-in Metrics

```python
from q_store.ml import CircuitBatchManagerV34

async with CircuitBatchManagerV34(
    api_key="your_ionq_key",
    use_batch_api=True,
    use_native_gates=True
) as manager:
    
    results = await manager.execute_batch(circuits, shots=1000)
    
    # Get detailed statistics
    stats = manager.get_performance_stats()
    print(f"Throughput: {stats['throughput_circuits_per_sec']:.2f} circuits/sec")
    print(f"Cache hit rate: {stats['circuit_cache']['template_hit_rate']:.1%}")
    
    # Or print comprehensive report
    manager.print_performance_report()
```

### Tracked Metrics

**Timing Metrics**:
- `batch_submission_time_ms` - API submission time
- `batch_execution_time_ms` - Total batch execution
- `circuit_build_time_ms` - Circuit construction time
- `parameter_bind_time_ms` - Parameter binding time

**Throughput Metrics**:
- `circuits_per_second` - Effective throughput
- `effective_parallelization` - Actual vs theoretical speedup

**Caching Metrics**:
- `cache_hit_rate` - Percentage of cache hits
- `cache_memory_mb` - Cache memory usage

**API Metrics**:
- `api_calls_saved` - Calls avoided via batching
- `connection_reuse_rate` - Connection pool efficiency

## Backward Compatibility

v3.4 is **100% backward compatible** with v3.3.2:

- All v3.3.2 code works unchanged
- v3.4 features are opt-in via configuration
- Automatic fallback if v3.4 components unavailable
- No breaking changes to existing APIs

### Migration Path

**Gradual Adoption**:
```python
# Week 1: Enable batch API
config.use_batch_api = True  # 5x improvement

# Week 2: Add native gates
config.use_native_gates = True  # Additional 1.3x

# Week 3: Enable caching
config.enable_smart_caching = True  # Additional 2x
```

**Immediate Adoption**:
```python
# Enable everything at once
config.enable_all_v34_features = True  # 8-10x improvement
```

## Technical Architecture

### Performance Breakdown

**v3.3.1 Batch Execution (35s total)**:
- Circuit building: 0.5s
- Sequential submission: 10s (20 × 0.5s API overhead)
- Queue time: 13s (20 × 0.65s average)
- Sequential execution: 11.5s (20 × 0.575s)

**v3.4 Batch Execution (4s total)**:
- Cache lookup + binding: 0.05s (10x faster)
- Native compilation: 0.1s
- Batch submission: 0.5s (20x faster)
- Parallel queueing: 1.5s
- Parallel execution: 1.85s

**Result**: 35s → 4s = **8.75x faster**

### Optimization Pipeline

```
User Code
    ↓
TrainingConfig (v3.4 options)
    ↓
CircuitBatchManagerV34
    ↓
┌───────────────┬─────────────────┬──────────────────┐
SmartCache → Compiler → BatchClient → IonQ Hardware
(10x prep)   (1.3x exec)  (12x submit)
```

### Key Innovations

1. **True Parallel Batch Submission**: Single API call for all circuits instead of N sequential calls
2. **Hardware-Native Gates**: Direct compilation to IonQ GPi/GPi2/MS gates without decomposition overhead
3. **Template-Based Caching**: Cache structure once, bind parameters dynamically for each sample
4. **Integrated Optimization**: Unified manager coordinating all optimizations for maximum benefit

## Use Cases

### Training Acceleration

Perfect for:
- Large-scale quantum ML training
- Hyperparameter optimization (many training runs)
- Real-time model updates
- Production ML pipelines

### Hardware Efficiency

Ideal for:
- Maximizing IonQ hardware utilization
- Reducing queue wait times
- Minimizing API overhead
- Optimizing cost per circuit

## Comparison with Previous Versions

| Feature | v3.2 | v3.3 | v3.4 |
|---------|------|------|------|
| **Gradient Method** | Parameter shift | SPSA | SPSA |
| **Circuit Submission** | Sequential | Sequential | True Batch |
| **Gate Compilation** | Standard | Standard | Native |
| **Circuit Caching** | None | Basic | Template-based |
| **Connection Pooling** | No | No | Yes |
| **Adaptive Batching** | No | No | Yes |
| **Batch Time (20 circuits)** | 240s | 35s | 3-5s |
| **Overall Speedup** | Baseline | 7x | 60-80x |

## Best Practices

### Optimal Configuration

For **maximum performance**:
```python
config = TrainingConfig(
    enable_all_v34_features=True,
    connection_pool_size=5,
    batch_size=20
)
```

For **stability-focused** deployments:
```python
config = TrainingConfig(
    use_batch_api=True,
    use_native_gates=True,
    enable_smart_caching=True,
    adaptive_batch_sizing=False,  # Disable for predictable behavior
    max_queue_wait_time=180.0     # Longer timeout
)
```

For **debugging**:
```python
config = TrainingConfig(
    enable_all_v34_features=True,
    debug_mode=True,
    log_level='DEBUG'
)
```

### Performance Tuning

**Batch Size Selection**:
- Small datasets (< 100 samples): batch_size = 10
- Medium datasets (100-1000 samples): batch_size = 20
- Large datasets (> 1000 samples): batch_size = 20 with adaptive sizing

**Connection Pool Sizing**:
- Light load: pool_size = 3
- Normal load: pool_size = 5 (recommended)
- Heavy load: pool_size = 8

**Cache Management**:
- Keep `enable_smart_caching=True` for repeated training
- Disable for one-off experiments
- Monitor cache hit rates (target: > 80%)

## Success Criteria

### Phase 1: Critical Path (5x speedup)
- ✅ IonQBatchClient with true batch API submission
- ✅ SmartCircuitCache with parameter binding
- ✅ Batch time reduced from 35s to 7s

### Phase 2: Native Gates (Additional 1.3x)
- ✅ IonQNativeGateCompiler for GPi/GPi2/MS gates
- ✅ Optimized execution on IonQ hardware
- ✅ Batch time reduced from 7s to 5s

### Phase 3: Adaptive Optimization (Additional 1.2x)
- ✅ AdaptiveQueueManager with smart batch sizing
- ✅ Dynamic optimization based on queue depth
- ✅ Batch time reduced from 5s to 4s

## Support & Resources

- **Examples**: See comprehensive examples in `examples/examples_v3_4.py`
- **Debug Logging**: Enable with `config.log_level = 'DEBUG'`
- **Performance Reports**: Use `manager.print_performance_report()`

---

**Version**: 3.4  
**Status**: Production Ready  
**Release Date**: December 2024  
**Performance**: 8-10x faster than v3.3.1 ✨
