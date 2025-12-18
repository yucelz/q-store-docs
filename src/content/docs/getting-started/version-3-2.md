---
title: Q-Store v3.2 (Retired)
description: Complete ML Training Capabilities for Quantum-Native Database
---

:::caution[Retired Version]
This version is retired. Please upgrade to [v3.5](/getting-started/version-3-5) better performance and production-ready optimizations.
:::

## Overview

Q-Store v3.2 introduces **complete machine learning training capabilities** with full hardware abstraction, enabling quantum neural networks that work seamlessly across simulators and quantum hardware.

## What's New in v3.2

### Core ML Components

#### Quantum Neural Network Layers
- **QuantumLayer**: Variational quantum circuit layer with trainable parameters
- **QuantumConvolutionalLayer**: Sliding window quantum convolution
- **QuantumPoolingLayer**: Dimensionality reduction via quantum measurements
- 3 rotation gates (RX, RY, RZ) per qubit per layer
- Multiple entanglement patterns (linear, circular, full)

#### Gradient Computation
- **Parameter Shift Rule**: Exact gradient computation for quantum circuits
- **Finite Difference**: Fallback gradient method
- **Natural Gradients**: Quantum Fisher information optimization
- Parallel gradient computation for efficiency
- Stochastic gradient estimation

#### Data Encoding
- **Amplitude Encoding**: N-dimensional data → log₂N qubits (exponential compression)
- **Angle Encoding**: Features mapped to rotation angles
- **Basis Encoding**: Binary feature representation
- **ZZ Feature Map**: Second-order Pauli expansion

#### Training Infrastructure
- **QuantumTrainer**: Complete training orchestration
- Adam optimizer with gradient clipping
- SGD with momentum
- Checkpoint management (save/load)
- Training metrics tracking
- Validation loop support

## Quick Start

### Installation

```bash
pip install q-store>=3.2.0
```

### Basic Training Example

```python
from q_store.core import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    BackendManager
)
from q_store.core.data_encoder import QuantumDataEncoder

# Configure training
config = TrainingConfig(
    pinecone_api_key="your-api-key",
    quantum_sdk="mock",  # or "cirq", "qiskit", "ionq"
    learning_rate=0.01,
    epochs=10,
    batch_size=5,
    n_qubits=4
)

# Initialize backend
backend_manager = BackendManager(config)
backend = backend_manager.get_backend("mock_ideal")

# Create model
model = QuantumModel(
    input_dim=4,
    output_dim=2,
    n_layers=2,
    backend=backend
)

# Prepare data
encoder = QuantumDataEncoder(n_qubits=4)
encoded_data = await encoder.encode_batch(training_data, method="amplitude")

# Train
trainer = QuantumTrainer(config, backend_manager)
history = await trainer.train(model, data_loader)

print(f"Final loss: {history['losses'][-1]:.4f}")
```

## Key Features

### 1. Hardware Abstraction

Train once, run anywhere:

```python
# Train on simulator
config.quantum_sdk = "mock"
await trainer.train(model, train_loader)

# Fine-tune on real quantum hardware
config.quantum_sdk = "ionq"
await trainer.train(model, train_loader, epochs=5)
```

### 2. Transfer Learning

Pre-train and fine-tune with parameter freezing:

```python
# Pre-train on task A
await trainer.train(model, task_a_loader)

# Freeze early layers
model.quantum_layer.freeze_parameters([0, 1, 2, 3])

# Fine-tune on task B
config.learning_rate = 0.001
await trainer.train(model, task_b_loader)
```

### 3. Multiple Encoding Strategies

Choose the best encoding for your data:

```python
# Amplitude encoding (exponential compression)
encoded = await encoder.encode(data, method="amplitude")
# 64-dim data → 6 qubits

# Angle encoding (direct feature mapping)
encoded = await encoder.encode(data, method="angle")

# ZZ Feature Map (second-order interactions)
encoded = await encoder.encode(data, method="zz_feature_map")
```

### 4. Advanced Gradient Methods

```python
from q_store.core.gradient_computer import (
    QuantumGradientComputer,
    NaturalGradientComputer
)

# Standard parameter shift
grad_computer = QuantumGradientComputer(backend)
gradients = await grad_computer.compute_gradients(circuit, params)

# Natural gradients (faster convergence)
nat_grad = NaturalGradientComputer(backend)
gradients = await nat_grad.compute_natural_gradient(circuit, params, loss)
```


## Performance Characteristics

### Circuit Execution Costs

```python
# Forward pass: 1 circuit execution
# Gradient (N params): 2N circuit executions
# Total per training step: 1 + 2N executions

# Example: 4 qubits, 2 layers
# Parameters: 4 qubits × 3 gates × 2 layers = 24 params
# Cost per step: 1 + (2 × 24) = 49 circuit executions
```

### Optimization Strategies

1. **Gradient Batching**: Average gradients over multiple samples
2. **Stochastic Gradients**: Random parameter subset selection
3. **Circuit Caching**: Reuse compiled circuits
4. **Backend Selection**: Choose simulator vs QPU based on needs

### Scalability

Tested configurations:
- **Qubits**: 2-8 qubits
- **Parameters**: 6-48 trainable parameters
- **Batch Size**: 2-10 samples
- **Training Time**: ~45 seconds for 5 epochs (mock backend)

## Examples

### 1. Basic Quantum Neural Network

```python
# Simple QNN on synthetic data
model = QuantumModel(input_dim=4, output_dim=2, n_layers=2)
await trainer.train(model, data_loader)
```

### 2. Quantum Autoencoder

```python
# Dimensionality reduction
model = QuantumModel(
    input_dim=8,
    output_dim=4,  # Compress to 4 dimensions
    n_layers=3
)
reconstructed = await model.forward(input_data)
```

### 3. Pattern Recognition

```python
# Train classifier
config.n_qubits = 8
model = QuantumModel(input_dim=8, output_dim=3, n_layers=4)
await trainer.train(model, pattern_data)

# Predict
predictions = await model.forward(test_patterns)
```

### 4. Multi-Backend Training

```python
# Compare performance across backends
backends = ["mock_ideal", "mock_noisy", "cirq"]
results = {}

for backend_name in backends:
    backend = backend_manager.get_backend(backend_name)
    model = QuantumModel(4, 2, 2, backend)
    history = await trainer.train(model, data_loader)
    results[backend_name] = history
```

## Advanced Features

### Checkpoint Management

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state': model.get_state(),
    'optimizer_state': optimizer.state,
    'loss': current_loss
}
trainer.save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch}.pkl")

# Load checkpoint
checkpoint = trainer.load_checkpoint("checkpoint_epoch_5.pkl")
model.load_state(checkpoint['model_state'])
```

### Custom Training Loop

```python
for epoch in range(config.epochs):
    epoch_loss = 0.0
    
    for batch in data_loader:
        # Forward pass
        predictions = await model.forward(batch['inputs'])
        
        # Compute loss
        loss = loss_function(predictions, batch['targets'])
        
        # Compute gradients
        gradients = await grad_computer.compute_gradients(
            model.quantum_layer.circuit,
            model.quantum_layer.parameters
        )
        
        # Update parameters
        optimizer.step(model.quantum_layer.parameters, gradients)
        
        epoch_loss += loss
    
    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
```

### Gradient Clipping

```python
config = TrainingConfig(
    learning_rate=0.01,
    gradient_clip_value=1.0,  # Clip gradients to [-1, 1]
    gradient_clip_norm=2.0     # Clip gradient norm
)
```


### Backward Compatibility


### Quick Start Test

```bash
# Run quick start demonstration
python quickstart_v3_2.py
```

Output:
```
Training quantum model (5 epochs, 50 samples, 4 qubits)
Epoch 1/5: Loss = 0.6234
Epoch 2/5: Loss = 0.4891
Epoch 3/5: Loss = 0.3782
Epoch 4/5: Loss = 0.3351
Epoch 5/5: Loss = 0.3139

Final loss: 0.3139
Gradient norm: 0.0236
Training time: ~45 seconds ✅
```

## Production Considerations

### Hardware Selection

```python
# Development: Use mock backend (free, fast)
config.quantum_sdk = "mock"

# Testing: Use noisy simulator (realistic)
backend = backend_manager.get_backend("mock_noisy")

# Production: Use quantum hardware (when needed)
config.quantum_sdk = "ionq"
backend = backend_manager.get_backend("ionq_qpu")
```

### Cost Estimation

```python
# Estimate training costs before running
n_params = model.count_parameters()
circuits_per_step = 1 + (2 * n_params)
total_circuits = circuits_per_step * len(data_loader) * config.epochs

estimated_cost = total_circuits * cost_per_circuit
print(f"Estimated cost: ${estimated_cost:.2f}")
```

### Error Handling

```python
try:
    history = await trainer.train(model, data_loader)
except QuantumCircuitError as e:
    logger.error(f"Circuit execution failed: {e}")
    # Fallback to simulator
    backend = backend_manager.get_backend("mock_ideal")
    history = await trainer.train(model, data_loader)
```

## Next Steps

- Explore [Architecture Overview](/concepts/architecture)
- Learn about [Quantum Principles](/concepts/quantum-principles)
- Review [Performance Optimization](/advanced/performance)
- Check [Production Guide](/production/monitoring)

## Resources

### Documentation
- Full API Reference in code docstrings
- 6 comprehensive examples in `examples_v3_2.py`
- Architecture guide in project docs

### Support
- GitHub Issues: Report bugs and feature requests
- Community: Join discussions on quantum ML
- Examples: Working code in `examples/` directory

## Roadmap

### v3.3 (Planned)
- Quantum federated learning
- Quantum continual learning
- Advanced error mitigation
- Multi-QPU orchestration

### v4.0 (Vision)
- Neural architecture search
- Automated circuit optimization
- Real-time training
- Framework integrations (PennyLane, TensorFlow Quantum)

---

**Status**: ✅ Complete and Production-Ready

**Total Implementation**: 1,842 lines of ML code, fully tested and documented
