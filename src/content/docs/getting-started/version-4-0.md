---
title: Version 4.0 - The Quantum ML Revolution
description: Q-Store v4.0 represents a fundamental transformation - bringing industry-standard ML APIs to quantum computing while maintaining our unique hardware optimization advantages.
---

:::caution[Design Phase]
**Version 4.0.0 is currently in the design phase** and scheduled for production. This documentation describes the planned architecture and features.
:::

## 🚀 The Big Announcement

### From Custom Framework to Industry Standard

Q-Store v4.0 represents a **fundamental architectural transformation** Quantum's proven patterns while maintaining our unique advantages in real quantum hardware optimization and quantum database capabilities.

**Core Philosophy**: *"Make quantum ML as easy as classical ML, but optimized for real quantum hardware."*

This is not just an update - it's a complete reimagining of how quantum machine learning should work in production environments.

**GitHub Discussions**: Share your thoughts on the v4.0 design 
  [![Discussions](https://img.shields.io/badge/GitHub-Discussions-blue?logo=githubdiscussions)](https://github.com/yucelz/q-store/discussions/5)

## 🎯 What Changes in v4.0

| Aspect | v3.5 (Current) | v4.0 (New) |
|--------|----------------|------------|
| **API** | Custom training loop | Keras/PyTorch standard API |
| **Integration** | Standalone framework | TensorFlow + PyTorch plugins |
| **Circuits** | Cirq or Qiskit | Unified representation |
| **Distributed** | Manual orchestration | Standard strategies (TF/PyTorch) |
| **Simulation** | IonQ + local | qsim + Lightning + IonQ |
| **Gradients** | SPSA only | Multiple methods |
| **Target Users** | Quantum researchers | **ML practitioners + Quantum researchers** |

### Key Innovations (Unique to Q-Store v4.0)

1. **🔄 Dual-Framework Support**: Both TensorFlow AND PyTorch (TFQ is TensorFlow-only)
2. **⚛️ IonQ Hardware Optimization**: Native gates, cost tracking, queue management
3. **💾 Quantum Database**: Integration with Pinecone for quantum state management
4. **🎯 Smart Backend Routing**: Auto-select optimal backend based on cost/performance
5. **💰 Production Cost Optimization**: Budget-aware training with automatic fallback

### Performance Targets

| Workload | v3.5 Actual | v4.0 Target | Method |
|----------|-------------|-------------|---------|
| Fashion MNIST (3 epochs) | 17.5 min | **5-7 min** | qsim + optimization |
| Circuits/second | 0.57 | **3-5** | GPU acceleration |
| Multi-node scaling | N/A | **0.8-0.9 efficiency** | Standard distributed training |
| IonQ hardware | N/A | **2x vs TFQ** | Native gates + optimization |

## 💡 The Strategy: Best of Both Worlds

```
  TensorFlow Quantum          Q-Store v3.5
  ────────────────            ────────────
  ✓ Keras API                 ✓ IonQ Native Gates  
  ✓ MultiWorker Scale         ✓ Cost Optimization
  ✓ qsim Simulator            ✓ Multi-SDK Support
  ✓ Standard Patterns         ✓ Quantum Database
            ↓                         ↓
            └─────────┬───────────────┘
                      ↓
            ┌─────────────────────┐
            │  Q-Store v4.0       │
            │                     │
            │  Best of Both       │
            │  Worlds             │
            └─────────────────────┘
```

## 📊 Feature Comparison Matrix

| Feature | TFQ | Q-Store v3.5 | Q-Store v4.0 (Planned) |
|---------|-----|--------------|------------------------|
| **Framework Integration** | ✅ TensorFlow | ❌ Custom | ✅ TensorFlow + PyTorch |
| **Keras API** | ✅ Native | ❌ No | ✅ Yes |
| **Circuit Framework** | Cirq Only | Cirq + Qiskit | ✅ Unified (all frameworks) |
| **Distributed Training** | ✅ MultiWorker | ⚠️ Manual | ✅ Standard (TF + PyTorch) |
| **Kubernetes Support** | ✅ tf-operator | ❌ No | ✅ Both tf-operator + PyTorch |
| **TensorBoard** | ✅ Native | ⚠️ Custom | ✅ Native |
| **Gradient Methods** | ✅ Multiple | ⚠️ SPSA only | ✅ Multiple |
| **IonQ Native Gates** | ❌ No | ✅ Yes | ✅ Yes |
| **IonQ Hardware** | ⚠️ Via backend | ✅ First-class | ✅ First-class |
| **GPU Simulation** | ⚠️ Limited | ❌ No | ✅ Yes (Lightning) |
| **State Vector Sim** | ✅ qsim | ⚠️ Local | ✅ qsim + Lightning |
| **Database Integration** | ❌ No | ✅ Pinecone | ✅ Pinecone |
| **Quantum State Mgmt** | ❌ No | ✅ Yes | ✅ Enhanced |
| **Multi-Backend** | ❌ No | ⚠️ Manual | ✅ Auto-routing |
| **Cost Optimization** | ❌ No | ⚠️ Basic | ✅ Advanced |
| **Scale Proven** | ✅ 10K+ CPUs | ❌ Unknown | 🎯 Target |

**Legend**: ✅ Yes, ⚠️ Partial, ❌ No, 🎯 Target for v4.0

## 🆕 What's New in v4.0

### 1. Native TensorFlow & PyTorch Integration

Use quantum layers just like classical layers with familiar APIs:

#### TensorFlow/Keras Example

```python
import tensorflow as tf
from q_store.tf import QuantumLayer

# Build a hybrid model with Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    QuantumLayer(
        circuit_fn=my_quantum_circuit,
        n_qubits=4,
        backend='qsim'  # GPU-accelerated simulator
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Standard Keras compilation and training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

#### PyTorch Example

```python
import torch
import torch.nn as nn
from q_store.torch import QuantumLayer

class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = nn.Linear(28*28, 32)
        self.quantum = QuantumLayer(
            circuit_fn=my_quantum_circuit,
            n_qubits=4,
            backend='lightning.gpu'  # PennyLane GPU simulator
        )
        self.output = nn.Linear(4, 10)
    
    def forward(self, x):
        x = torch.relu(self.classical(x))
        x = self.quantum(x)
        return self.output(x)

# Standard PyTorch training loop
model = HybridQNN()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
```

### 2. Unified Circuit Representation

Write circuits once, run anywhere:

```python
from q_store import QuantumCircuit

# Unified circuit builder
circuit = QuantumCircuit(n_qubits=4)
circuit.h(0)
circuit.cx(0, 1)
circuit.rx(1, param='theta')
circuit.measure_all()

# Automatically converts to:
# - Cirq circuits for qsim backend
# - Qiskit circuits for IBM/Aer backend
# - IonQ native gates for IonQ hardware
# - PennyLane templates for Lightning backend
```

### 3. Advanced Gradient Computation

Multiple gradient methods for different use cases:

```python
from q_store import GradientConfig

# Parameter-shift rule (exact gradients)
config = GradientConfig(method='parameter_shift')

# Finite differences (fast approximation)
config = GradientConfig(method='finite_diff', epsilon=0.01)

# SPSA (high-dimensional optimization)
config = GradientConfig(method='spsa', samples=100)

# Adjoint method (efficient for simulators)
config = GradientConfig(method='adjoint')
```

### 4. Smart Backend Routing

Automatic backend selection based on workload:

```python
from q_store import BackendRouter

router = BackendRouter(
    preferences={
        'cost': 0.3,        # 30% weight on cost
        'speed': 0.5,       # 50% weight on speed
        'accuracy': 0.2     # 20% weight on accuracy
    },
    budget_limit=100.00,    # Max $100 for this job
    fallback_strategy='simulation'  # Use simulator if budget exceeded
)

# Router automatically selects:
# - qsim for small circuits (fast, free)
# - Lightning GPU for medium circuits (very fast, free)
# - IonQ simulator for testing (moderate cost)
# - IonQ hardware for final runs (high cost, exact results)
```

### 5. Production-Ready Distributed Training

#### TensorFlow MultiWorkerMirroredStrategy

```python
import tensorflow as tf
from q_store.tf import QuantumLayer

# Standard TensorFlow distributed strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        QuantumLayer(circuit_fn=my_circuit, n_qubits=4),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )

# Automatically distributes across workers
model.fit(train_dataset, epochs=10)
```

#### PyTorch DistributedDataParallel

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from q_store.torch import QuantumLayer

# Standard PyTorch distributed setup
dist.init_process_group(backend='nccl')

model = HybridQNN().to(device)
model = DDP(model, device_ids=[local_rank])

# Standard distributed training
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
```

### 6. Enhanced Quantum Database Integration

Persistent quantum state management with improved performance:

```python
from q_store import QuantumStateDB

db = QuantumStateDB(
    provider='pinecone',
    index_name='quantum_states',
    # New in v4.0: Automatic compression
    compression='zstd',
    compression_level=3
)

# Store quantum states with metadata
db.store_state(
    state_id='training_checkpoint_epoch_5',
    state_vector=quantum_state,
    metadata={
        'epoch': 5,
        'loss': 0.342,
        'accuracy': 0.876,
        'circuit_depth': 12
    }
)

# Similarity search for quantum states
similar_states = db.find_similar(
    query_state=current_state,
    top_k=5,
    filter={'accuracy': {'$gte': 0.85}}
)
```

### 7. IonQ Native Gate Optimization

Optimized compilation for IonQ hardware:

```python
from q_store.ionq import IonQOptimizer

optimizer = IonQOptimizer(
    target_hardware='ionq.aria',
    optimization_level=3,
    # New in v4.0: Cost-aware gate selection
    minimize_cost=True,
    max_cost_per_shot=0.01
)

# Automatically converts to native gates
optimized_circuit = optimizer.optimize(circuit)

# Detailed cost estimation before execution
cost_estimate = optimizer.estimate_cost(
    circuit=optimized_circuit,
    shots=1000
)

print(f"Estimated cost: ${cost_estimate.total_cost:.2f}")
print(f"Gate count: {cost_estimate.native_gate_count}")
print(f"Circuit depth: {cost_estimate.optimized_depth}")
```

### 8. GPU-Accelerated Simulation

Leverage modern GPU simulators for massive speedups:

```python
from q_store import SimulatorConfig

# qsim (Google's fast simulator)
config = SimulatorConfig(
    backend='qsim',
    device='GPU',
    precision='single'  # Faster, 32-bit precision
)

# PennyLane Lightning (highly optimized)
config = SimulatorConfig(
    backend='lightning.gpu',
    device='cuda:0',
    batch_size=256  # Process 256 circuits in parallel
)

# Automatic selection based on circuit size
config = SimulatorConfig(
    backend='auto',
    prefer_gpu=True,
    # Use GPU for >12 qubits, CPU otherwise
    gpu_threshold=12
)
```

### 9. Native TensorBoard Support

Full integration with TensorBoard for experiment tracking:

```python
from q_store.tf import QuantumLayer
import tensorflow as tf

# TensorBoard automatically tracks quantum metrics
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# View quantum-specific metrics in TensorBoard:
# - Circuit execution time per batch
# - Gradient estimation variance
# - Backend utilization
# - Cost per epoch (for hardware backends)
```

### 10. Advanced Cost Optimization

Budget-aware training with automatic fallback:

```python
from q_store import CostOptimizer

optimizer = CostOptimizer(
    total_budget=500.00,  # $500 total budget
    daily_limit=50.00,    # Max $50 per day
    strategy='adaptive',   # Adjust based on progress
    
    # New in v4.0: Intelligent fallback chain
    fallback_chain=[
        {'backend': 'ionq.aria', 'max_cost_per_batch': 1.00},
        {'backend': 'ionq.simulator', 'max_cost_per_batch': 0.10},
        {'backend': 'lightning.gpu', 'max_cost_per_batch': 0.00},
        {'backend': 'qsim', 'max_cost_per_batch': 0.00}
    ]
)

# Training automatically switches backends when budget exceeded
trainer = QuantumTrainer(
    model=model,
    cost_optimizer=optimizer,
    auto_checkpoint=True  # Checkpoint before switching backends
)

trainer.fit(x_train, y_train, epochs=20)

# Detailed cost report
print(optimizer.get_cost_report())
```

## 🔄 Migration from v3.5

### Breaking Changes

:::danger[Breaking API Changes]
Version 4.0 introduces breaking changes to maintain consistency with industry-standard ML frameworks. Please review the migration guide carefully.
:::

#### 1. Training Loop API

**v3.5:**
```python
from q_store import QuantumTrainer, TrainingConfig

config = TrainingConfig(
    learning_rate=0.01,
    epochs=10,
    backend='ionq'
)

trainer = QuantumTrainer(config)
trainer.train(model, x_train, y_train)
```

**v4.0:**
```python
import tensorflow as tf
from q_store.tf import QuantumLayer

# Use standard Keras API
model = tf.keras.Sequential([
    QuantumLayer(circuit_fn=my_circuit, n_qubits=4)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10)
```

#### 2. Circuit Definition

**v3.5:**
```python
from q_store import QuantumCircuit

circuit = QuantumCircuit(backend='cirq')
circuit.add_gate('H', 0)
circuit.add_gate('CNOT', [0, 1])
```

**v4.0:**
```python
from q_store import QuantumCircuit

# Backend-agnostic circuit
circuit = QuantumCircuit(n_qubits=4)
circuit.h(0)
circuit.cx(0, 1)

# Backend selected at execution time
```

#### 3. Backend Configuration

**v3.5:**
```python
config = TrainingConfig(
    backend='ionq',
    ionq_api_key='your_key'
)
```

**v4.0:**
```python
from q_store import configure

# Global configuration
configure(
    ionq_api_key='your_key',
    default_backend='auto',  # Smart selection
    cache_dir='~/.qstore/cache'
)
```

### Migration Checklist

- [ ] Update import statements (`q_store.tf` or `q_store.torch`)
- [ ] Replace `QuantumTrainer` with Keras/PyTorch training loops
- [ ] Update circuit definitions to unified API
- [ ] Configure backends using new `configure()` function
- [ ] Update distributed training to use TF/PyTorch strategies
- [ ] Review gradient computation methods (new options available)
- [ ] Test with `qsim` or `lightning.gpu` simulators (recommended)
- [ ] Update monitoring to use TensorBoard instead of custom logging

## 🎯 Why This Matters

### For ML Practitioners

- **Zero Learning Curve**: If you know Keras or PyTorch, you already know Q-Store v4.0
- **Standard Tooling**: Use familiar tools like TensorBoard, distributed strategies, and callbacks
- **Production Ready**: Deploy quantum models using the same infrastructure as classical models

### For Quantum Researchers

- **Hardware Optimization**: Best-in-class IonQ native gate compilation
- **Multi-Backend**: Seamlessly switch between simulators and real hardware
- **Cost Control**: Run experiments within budget constraints
- **State Management**: Persist and analyze quantum states at scale

### For Organizations

- **Reduced Risk**: Built on proven TensorFlow and PyTorch architectures
- **Cost Transparency**: Track quantum computing costs alongside traditional infrastructure
- **Scalability**: Leverage existing distributed training infrastructure


- **Feature Requests**: Suggest features for the final release
- **Early Testing**: Join the beta program (weeks 7-8)
- **Documentation**: Help improve examples and guides


---

**Ready for the future of quantum machine learning?** Star us on [GitHub](https://github.com/yucelz/q-store-examples) and join the discussion!
