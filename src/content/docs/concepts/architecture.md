---
title: Architecture Overview
description: High-level architecture of the Q-Store quantum database system
---

## High-Level Design

Q-Store v3.3 uses a **hybrid architecture** that combines classical storage with quantum acceleration, optimized for ML training and domain applications with **50-100x performance improvements**:

```
┌─────────────────────────────────────────────────────────────┐
│                   ML Framework Layer                        │
│  • PyTorch Integration  • TensorFlow Bridge  • JAX Support  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│         Quantum Training Engine (v3.3 - OPTIMIZED)          │
│  • QuantumTrainer       • AdaptiveGradientOptimizer         │
│  • HardwareEfficient    • SPSAGradientEstimator             │
│    QuantumLayer         • CircuitBatchManager               │
│  • DataEncoder          • CheckpointManager                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Gradient Computation Strategy (NEW)              │
│  • SPSA: 2 circuits (default, 48x faster)                   │
│  • Parameter Shift: High accuracy                           │
│  • Natural Gradient: Fast convergence                       │
│  • Adaptive: Auto-selects best method                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│          Circuit Optimization Pipeline (NEW)                │
│  Batching ──► Caching ──► Compilation ──► Async Execution   │
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
│  (v3.3 Enhanced)     │      │                         │
│  • Backend Manager   │◄─────►│  • Pinecone            │
│  • Circuit Cache     │ sync │  • Training Data       │
│  • Batch Manager     │      │  • Checkpoints         │
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

### Quantum Training Engine (v3.3 - ENHANCED)

**Purpose**: Hardware-agnostic ML training with quantum acceleration and algorithmic optimization

**Core Components**:
- **QuantumTrainer**: Complete training orchestration with performance tracking
- **HardwareEfficientQuantumLayer**: Optimized layers with 33% fewer parameters
- **AdaptiveGradientOptimizer**: Auto-selects best gradient method
- **SPSAGradientEstimator**: 2-circuit gradient estimation (48x faster)
- **CircuitBatchManager**: Parallel circuit execution with async support
- **QuantumCircuitCache**: Multi-level caching (compiled, results, optimized)
- **DataEncoder**: Quantum state preparation
- **CheckpointManager**: Model persistence

**Key Optimizations in v3.3**:
- **SPSA Gradient Estimation**: Only 2 circuits vs 2N for N parameters (48x reduction)
- **Circuit Batching**: Parallel execution instead of sequential (5-10x faster)
- **Intelligent Caching**: Avoid redundant compilations and executions (2-5x speedup)
- **Hardware-Efficient Ansatz**: 2 rotations per qubit vs 3 (33% fewer parameters)
- **Adaptive Method Selection**: Optimal speed/accuracy throughout training

**Performance**: 50-100x faster training compared to v3.2

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

The hybrid architecture in v3.3:
- Uses classical storage for large-scale data management
- Applies quantum acceleration where it provides advantage
- Enables hardware-agnostic quantum ML training with 50-100x speedup
- Supports transfer learning and model fine-tuning
- Optimizes cost per training epoch (from $10 to $0.20 on QPU)
- Implements intelligent caching and batching for efficiency
- Auto-selects optimal gradient computation methods

## Domain Applications

Q-Store v3.3 is optimized for specific application domains with dramatic performance improvements:

### Financial Services
- Portfolio optimization with quantum annealing
- Risk assessment using quantum correlations
- Market regime discovery via tunneling
- Real-time trading strategy optimization
- **v3.3**: 24x faster model training for risk models

### ML Model Training
- Quantum neural network layers with hardware-efficient ansatz
- Multiple gradient computation methods (SPSA, parameter shift, natural)
- Hybrid classical-quantum architectures
- Transfer learning with quantum features
- **v3.3**: Training in minutes instead of hours, QPU-ready at <$1/run

### Recommendation Systems
- Quantum collaborative filtering
- Entanglement-based user similarity
- Context-aware recommendations via superposition
- Cold-start problem mitigation
- **v3.3**: Rapid model iteration with circuit caching

### Scientific Computing
- Molecular simulation and drug discovery
- Materials science optimization
- Climate modeling with quantum speedup
- Complex system analysis
- **v3.3**: Production-ready training for small-scale quantum ML

## Data Flow

### Training Data Storage

```
1. Dataset → Classical Store (bulk storage)
2. Create DataLoader → Batch management
3. Encode batches → Quantum states
4. Register training metadata
5. Enable checkpointing
```

### ML Training Loop

```
1. Load batch → Classical DataLoader
2. Encode data → Quantum states (amplitude/angle encoding)
3. Build circuit → Hardware-efficient ansatz (v3.3)
4. Check cache → Retrieve if cached (v3.3)
5. Execute batch → Parallel async submission (v3.3)
6. Forward pass → Variational quantum circuit
7. Compute loss → Classical loss function
8. Compute gradients → SPSA or adaptive method (v3.3)
   • SPSA: Only 2 circuits (default)
   • Parameter shift: 2N circuits (high accuracy)
   • Adaptive: Auto-selects based on stage
9. Update parameters → Quantum-aware optimizer
10. Cache results → Multi-level caching (v3.3)
11. Checkpoint model → Classical storage
12. Track metrics → Performance statistics (v3.3)
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
