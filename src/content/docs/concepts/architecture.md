---
title: Architecture Overview
description: High-level architecture of the Q-Store quantum database system
---

## High-Level Design

Q-Store v3.2 uses a **hybrid architecture** that combines classical storage with quantum acceleration, optimized for ML training and domain applications:

```
┌─────────────────────────────────────────────────────────────┐
│                   ML Framework Layer                        │
│  • PyTorch Integration  • TensorFlow Bridge  • JAX Support  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Quantum Training Engine (v3.2)                 │
│  • QuantumTrainer       • QuantumOptimizer                  │
│  • QuantumLayer         • GradientComputer                  │
│  • DataEncoder          • CheckpointManager                 │
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
│                      │      │                         │
│  • Backend Manager   │◄─────►│  • Pinecone            │
│  • Circuit Cache     │ sync │  • Training Data       │
│  • State Manager     │      │  • Checkpoints         │
└───────┬──────────────┘      └─────────────────────────┘
        │
┌───────▼──────────────┐
│  Quantum Backends    │
│  • Cirq/IonQ         │
│  • Qiskit/IonQ       │
│  • Mock Simulator    │
└──────────────────────┘
```

## Component Responsibilities

### Quantum Training Engine (NEW in v3.2)

**Purpose**: Hardware-agnostic ML training with quantum acceleration

- Quantum neural network layers with trainable parameters
- Quantum gradient computation using parameter shift rule
- Hybrid classical-quantum training pipelines
- Training data management and model checkpointing
- Integration with PyTorch, TensorFlow, and JAX

### Classical Component

**Purpose**: Bulk storage and training data management

- Stores training datasets and model checkpoints
- Handles embeddings and feature vectors
- Provides efficient batch loading for training
- Mature, reliable, cost-effective storage
- Options: Pinecone, pgvector, Qdrant

### Quantum Component

**Purpose**: Quantum-accelerated computation and ML processing

- Quantum state encoding and superposition
- Variational quantum circuits for ML
- Quantum gradient computation
- Entanglement-based feature learning
- Hardware-agnostic backend management (Cirq, Qiskit)

## Why Hybrid?

Current quantum computers (NISQ era) have limitations:
- Limited qubits (10-100 available for useful computation)
- Short coherence times (milliseconds)
- Costly execution per circuit

The hybrid architecture in v3.2:
- Uses classical storage for large-scale data management
- Applies quantum acceleration where it provides advantage
- Enables hardware-agnostic quantum ML training
- Supports transfer learning and model fine-tuning
- Optimizes cost per training epoch

## Domain Applications

Q-Store v3.2 is optimized for specific application domains:

### Financial Services
- Portfolio optimization with quantum annealing
- Risk assessment using quantum correlations
- Market regime discovery via tunneling
- Real-time trading strategy optimization

### ML Model Training
- Quantum neural network layers
- Gradient computation with parameter shift rule
- Hybrid classical-quantum architectures
- Transfer learning with quantum features

### Recommendation Systems
- Quantum collaborative filtering
- Entanglement-based user similarity
- Context-aware recommendations via superposition
- Cold-start problem mitigation

### Scientific Computing
- Molecular simulation and drug discovery
- Materials science optimization
- Climate modeling with quantum speedup
- Complex system analysis

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
3. Forward pass → Variational quantum circuit
4. Compute loss → Classical loss function
5. Compute gradients → Quantum parameter shift rule
6. Update parameters → Quantum-aware optimizer
7. Checkpoint model → Classical storage
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
