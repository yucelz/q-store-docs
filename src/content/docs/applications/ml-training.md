---
title: ML Model Training
description: Using Q-Store v4.0.0 for machine learning applications
---

Q-Store v4.0.0 provides quantum-enhanced database operations and PyTorch integration for hybrid classical-quantum machine learning.

## Core Use Cases

### 1. Hybrid Quantum-Classical Training

Use Q-Store's `QuantumLayer` for PyTorch models:

```python
import torch
from q_store import QuantumLayer, DatabaseConfig

config = DatabaseConfig(
    quantum_sdk='ionq',  # or 'mock' for development
    ionq_api_key="your-key"
)

# Create quantum layer
quantum_layer = QuantumLayer(
    n_qubits=4,
    depth=2,
    config=config
)

# Build hybrid model
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_1 = torch.nn.Linear(10, 4)
        self.quantum = quantum_layer
        self.classical_2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.classical_1(x)
        x = self.quantum(x)  # Quantum processing
        x = self.classical_2(x)
        return x

model = HybridModel()
```

**Performance:**
- 500 samples, 2 epochs, 4 qubits: 19.5 seconds
- Full gradient computation via parameter shift rule
- Hardware-agnostic (works with IonQ, Cirq, Qiskit, mock)

### 2. Training Data Selection with Superposition

Store training examples in multiple task contexts:

```python
from q_store import QuantumDatabase, DatabaseConfig

config = DatabaseConfig(
    enable_quantum=True,
    enable_superposition=True,
    pinecone_api_key="your-key"
)

ml_db = QuantumDatabase(config)

# Store example in superposition
ml_db.insert(
    id='example_1',
    vector=embedding,
    contexts=[
        ('classification', 0.6),
        ('regression', 0.3),
        ('clustering', 0.1)
    ]
)

# Query for classification task
classification_data = ml_db.query(
    vector=query_embedding,
    context='classification',
    top_k=100
)
```

**Benefits:**
- One dataset serves multiple tasks
- Context-aware data selection
- Exponential compression of task contexts

### 3. Hyperparameter Optimization with Tunneling

Use quantum tunneling to escape local optima:

```python
# Find optimal hyperparameters
best_config = ml_db.tunnel_search(
    query=current_model_state,
    barrier_threshold=0.8,
    tunneling_strength=0.6,
    top_k=20
)

# Tunneling finds globally better configurations
# that classical grid/random search miss
```

**Benefits:**
- Escape local optima in hyperparameter space
- Find globally optimal configurations
- Faster convergence than grid/random search

### 4. Similar Model Discovery

Find similar pre-trained models using quantum search:

```python
# Store model embeddings
ml_db.insert(
    id='bert_base',
    vector=model_embedding,
    metadata={'arch': 'transformer', 'params': '110M'}
)

# Find similar models with tunneling
similar_models = ml_db.tunnel_search(
    query=target_model_embedding,
    barrier_threshold=0.7,
    top_k=10
)
```

## Complete Example: Quantum-Enhanced Training

```python
import torch
from q_store import QuantumLayer, QuantumDatabase, DatabaseConfig
import numpy as np

# 1. Setup quantum database for training data
db_config = DatabaseConfig(
    enable_quantum=True,
    enable_superposition=True,
    pinecone_api_key=PINECONE_KEY
)

training_db = QuantumDatabase(db_config)

# 2. Store training data with contexts
for i, (data, label) in enumerate(training_data):
    training_db.insert(
        id=f'train_{i}',
        vector=data,
        contexts=[
            (f'class_{label}', 1.0)
        ],
        metadata={'label': label}
    )

# 3. Create hybrid model with quantum layer
model_config = DatabaseConfig(
    quantum_sdk='ionq',
    ionq_api_key=IONQ_KEY
)

class QuantumClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = torch.nn.Linear(784, 4)
        self.quantum = QuantumLayer(n_qubits=4, depth=2, config=model_config)
        self.decode = torch.nn.Linear(4, 10)

    def forward(self, x):
        x = self.encode(x)
        x = self.quantum(x)
        x = self.decode(x)
        return x

model = QuantumClassifier()

# 4. Train with standard PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()  # Quantum gradients computed automatically
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. Find similar models using quantum search
model_embedding = extract_embedding(model)
similar_models = training_db.tunnel_search(
    query=model_embedding,
    barrier_threshold=0.7,
    top_k=5
)
```

## Performance Characteristics

Based on v4.0.0 benchmarks:

| Operation | Time | Notes |
|-----------|------|-------|
| QuantumLayer forward pass | <1ms | 4 qubits, depth 2 |
| Gradient computation | ~2ms | Parameter shift rule |
| Training (500 samples, 2 epochs) | 19.5s | 4 qubits |
| Database query | ~60-100ms | With quantum enhancement |

## PyTorch Integration

### QuantumLayer API

```python
from q_store import QuantumLayer

layer = QuantumLayer(
    n_qubits=4,        # Number of qubits
    depth=2,           # Circuit depth
    config=config,     # Quantum backend config
    entanglement='linear'  # or 'circular', 'full'
)

# Use like any PyTorch layer
output = layer(input_tensor)  # Shape: (batch_size, n_qubits)
```

### Gradient Computation

Q-Store automatically computes gradients using parameter shift rule:

```python
# Standard PyTorch training loop
output = model(input)
loss = criterion(output, target)
loss.backward()  # Quantum gradients computed via parameter shift

optimizer.step()  # Update quantum parameters
```

## Best Practices

### 1. Choose Appropriate Qubit Count

- **2-4 qubits**: Fast training, good for prototyping
- **4-6 qubits**: Balanced expressivity and speed
- **6-8 qubits**: Maximum expressivity (slower)
- **8+ qubits**: Requires specialized hardware

### 2. Select Circuit Depth

- **Depth 1**: Fast but limited expressivity
- **Depth 2-4**: Recommended for most use cases
- **Depth 5+**: High expressivity but slower, more decoherence

### 3. Use Mock Mode for Development

```python
# Development with mock backend (fast, free)
config = DatabaseConfig(quantum_sdk='mock')
quantum_layer = QuantumLayer(n_qubits=4, depth=2, config=config)

# Production with IonQ (accurate, costs money)
config = DatabaseConfig(
    quantum_sdk='ionq',
    ionq_api_key="your-key"
)
```

### 4. Optimize Training Data Selection

```python
# Use quantum superposition for multi-task learning
for data in training_data:
    ml_db.insert(
        id=data['id'],
        vector=data['embedding'],
        contexts=data['task_contexts']  # Multiple tasks
    )

# Query selectively for each task
task_data = ml_db.query(
    vector=query,
    context='task_name',
    top_k=batch_size
)
```

## Limitations in v4.0.0

- **GPU Support**: Quantum layers return CPU tensors (GPU pending)
- **Qubit Scaling**: Optimized for 4-8 qubits
- **Mock Accuracy**: Random results (~10-20%), use for structure only
- **IonQ Accuracy**: 60-75% (NISQ hardware constraints)
- **Training Time**: Slower than classical (quantum overhead)

## When to Use Quantum ML

✅ **Use quantum layers for:**
- Parameter-efficient models (<500 parameters)
- Few-shot learning scenarios
- Exploring quantum advantages
- Research and experimentation

❌ **Stick with classical for:**
- High-accuracy requirements (>90%)
- Large-scale production models
- Real-time inference (<10ms)
- Standard classification tasks

## Debugging Quantum Training

### Check Quantum Layer Output

```python
# Verify quantum layer is working
quantum_layer = QuantumLayer(n_qubits=4, depth=2, config=config)
test_input = torch.randn(1, 4)
output = quantum_layer(test_input)

print(f"Output shape: {output.shape}")  # Should be (1, 4)
print(f"Output device: {output.device}")  # Currently: cpu
print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
```

### Monitor Gradient Flow

```python
# Check if gradients are flowing
model = QuantumClassifier()
output = model(test_input)
loss = output.sum()
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

## Next Steps

- Learn about [Financial Applications](/applications/financial) for quantum-enhanced trading
- Check [Recommendation Systems](/applications/recommendations) for personalization
- Explore [Scientific Computing](/applications/scientific) for molecular ML
- Review [Circuit Builder](/components/circuit-builder) for quantum circuit details
