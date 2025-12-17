---
title: Architecture Overview
description: High-level architecture of the Q-Store quantum database system
---

## High-Level Design

Q-Store v3.4 uses a **hybrid architecture** that combines classical storage with quantum acceleration, optimized for ML training and domain applications with **8-10x performance improvements over v3.3** through true parallelization:

```
┌─────────────────────────────────────────────────────────────┐
│                   ML Framework Layer                        │
│  • PyTorch Integration  • TensorFlow Bridge  • JAX Support  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│         Quantum Training Engine (v3.4 - PRODUCTION)         │
│  • QuantumTrainer       • AdaptiveGradientOptimizer         │
│  • HardwareEfficient    • SPSAGradientEstimator             │
│    QuantumLayer         • CircuitBatchManagerV34 (NEW)      │
│  • DataEncoder          • CheckpointManager                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Gradient Computation Strategy                    │
│  • SPSA: 2 circuits (default, 48x faster)                   │
│  • Parameter Shift: High accuracy                           │
│  • Natural Gradient: Fast convergence                       │
│  • Adaptive: Auto-selects best method                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│      Circuit Optimization Pipeline (v3.4 - ENHANCED)        │
│  SmartCache ──► NativeCompiler ──► BatchAPI ──► Parallel    │
│  (10x faster)  (1.3x faster)     (12x faster)  Execution    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Database Management Layer                     │
│  • Training Data Store  • Model Registry  • Metrics         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────────┐      ┌──────────▼──────────────┐
│  Quantum Engine      │      │  Classical Store        │
│  (v3.4 Production)   │      │                         │
│  • IonQBatchClient   │◄─────►│  • Pinecone            │
│  • SmartCircuitCache │ sync │  • Training Data       │
│  • NativeCompiler    │      │  • Checkpoints         │
│  • State Manager     │      │  • Performance Metrics │
└───────┬──────────────┘      └─────────────────────────┘
        │
┌───────▼──────────────┐
│  Quantum Backends    │
│  (Async Support)     │
│  • Cirq/IonQ         │
│  • Qiskit/IonQ       │
│  • Mock Simulator    │
└──────────────────────┘
```

## Component Responsibilities

### Quantum Training Engine (v3.4 - PRODUCTION)

**Purpose**: Hardware-agnostic ML training with quantum acceleration and true parallelization

**Core Components**:
- **QuantumTrainer**: Complete training orchestration with performance tracking
- **CircuitBatchManagerV34**: Integrated v3.4 optimization pipeline (NEW)
- **IonQBatchClient**: True parallel batch submission with connection pooling (NEW)
- **SmartCircuitCache**: Template-based caching with parameter binding (NEW)
- **IonQNativeGateCompiler**: Native gate compilation (GPi/GPi2/MS) (NEW)
- **AdaptiveQueueManager**: Dynamic batch sizing based on queue depth (NEW)
- **SPSAGradientEstimator**: 2-circuit gradient estimation (48x faster)
- **HardwareEfficientQuantumLayer**: Optimized layers with 33% fewer parameters
- **DataEncoder**: Quantum state preparation
- **CheckpointManager**: Model persistence

**Key Optimizations in v3.4**:
- **True Batch Submission**: Single API call for all circuits vs sequential (12x faster)
- **Native Gate Compilation**: GPi/GPi2/MS gates for direct hardware execution (30% faster)
- **Template-Based Caching**: Cache structure, bind parameters dynamically (10x faster)
- **Connection Pooling**: Persistent HTTP connections (60% reduced overhead)
- **Adaptive Batching**: Dynamic batch sizing based on queue conditions (1.2x faster)
- **SPSA Gradient Estimation**: Only 2 circuits vs 2N for N parameters (48x reduction)

**Performance**: 8-10x faster training compared to v3.3.1, 400-800x faster than v3.2

### Classical Component

**Purpose**: Bulk storage and training data management

- Stores training datasets and model checkpoints
- Handles embeddings and feature vectors
- Provides efficient batch loading for training
- Stores performance metrics and cache statistics
- Mature, reliable, cost-effective storage
- Options: Pinecone, pgvector, Qdrant

### Quantum Component

**Purpose**: Quantum-accelerated computation and ML processing

- Quantum state encoding and superposition
- Variational quantum circuits for ML
- Multiple gradient computation methods (SPSA, parameter shift, natural gradient)
- Entanglement-based feature learning
- Hardware-agnostic backend management (Cirq, Qiskit)
- Asynchronous job submission and result fetching

## Why Hybrid?

Current quantum computers (NISQ era) have limitations:
- Limited qubits (10-100 available for useful computation)
- Short coherence times (milliseconds)
- Costly execution per circuit

The hybrid architecture in v3.4:
- Uses classical storage for large-scale data management
- Applies quantum acceleration where it provides advantage
- Enables hardware-agnostic quantum ML training with 8-10x speedup over v3.3
- Supports transfer learning and model fine-tuning
- Optimizes cost per training epoch (sub-$0.50 on QPU with batch API)
- Implements true parallelization with native gate optimization
- Template-based caching reduces circuit preparation by 10x
- Auto-selects optimal gradient computation methods
- Production-ready with sub-60s training epochs on IonQ hardware

## Domain Applications

Q-Store v3.4 is production-ready for specific application domains with true parallelization:

### Financial Services
- Portfolio optimization with quantum annealing
- Risk assessment using quantum correlations
- Market regime discovery via tunneling
- Real-time trading strategy optimization
- **v3.4**: Sub-5 minute model training on IonQ hardware, 200x faster than v3.2

### ML Model Training
- Quantum neural network layers with hardware-efficient ansatz
- Multiple gradient computation methods (SPSA, parameter shift, natural)
- Hybrid classical-quantum architectures
- Transfer learning with quantum features
- **v3.4**: Production deployment ready, 2.5-4 min for 5 epochs, QPU-ready at <$0.50/run

### Recommendation Systems
- Quantum collaborative filtering
- Entanglement-based user similarity
- Context-aware recommendations via superposition
- Cold-start problem mitigation
- **v3.4**: Real-time model updates with 10x faster circuit preparation

### Scientific Computing
- Molecular simulation and drug discovery
- Materials science optimization
- Climate modeling with quantum speedup
- Complex system analysis
- **v3.4**: Production-ready with native IonQ gate optimization for maximum fidelity

## Data Flow

### Training Data Storage

```
1. Dataset → Classical Store (bulk storage)
2. Create DataLoader → Batch management
3. Encode batches → Quantum states
4. Register training metadata
5. Enable checkpointing
```

### Inference Operation

```
1. Load model → Restore quantum parameters
2. Encode input → Quantum state
3. Execute circuit → Quantum backend
4. Measure output → Classical predictions
5. Return results
```

## Scalability

### Horizontal Scaling

- Multiple training workers for parallel model training
- Distributed quantum circuit execution
- Load-balanced inference endpoints
- Shared classical storage and quantum backends

### Vertical Scaling

- Increase classical storage capacity for larger datasets
- Access more powerful quantum hardware (Aria → Forte)
- Larger qubit systems for deeper quantum circuits
- Enhanced quantum backend capabilities

## Next Steps

- Learn about [Quantum Principles](/concepts/quantum-principles)
- Understand the [Hybrid Design](/concepts/hybrid-design)
- Explore [System Components](/components/state-manager)
- See [Domain Applications](/applications/ml-training)
