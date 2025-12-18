---
title: Example Projects
description: Comprehensive collection of Q-Store examples demonstrating quantum database capabilities, ML training, and real-world applications.
---

## Overview

The [Q-Store Examples Repository](https://github.com/yucelz/q-store-examples) provides standalone example projects demonstrating Q-Store quantum database capabilities for machine learning training, financial applications, and more.

All examples support both **mock mode** (for safe testing without API calls) and **real backends** (Pinecone + IonQ) with flexible configuration options.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples

# Install in editable mode (recommended)
pip install -e .

# Or with optional ML dependencies
pip install -e ".[ml,data,dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Verify installation
python scripts/verify_installation.py
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east-1

# Optional (for quantum features)
IONQ_API_KEY=your_ionq_key
IONQ_TARGET=simulator

# ML Training (optional)
HUGGING_FACE_TOKEN=your_token
```

## Available Examples

### 1. Basic Example

**File**: `basic_example.py`

Demonstrates core Q-Store functionality including:
- Inserting vectors with quantum contexts
- Querying with superposition
- Creating entangled groups
- Quantum tunneling for exploration

```bash
python basic_example.py
```

### 2. Financial Example

**File**: `financial_example.py`

Financial data analysis with quantum features:
- Portfolio optimization
- Risk correlation analysis
- Market regime detection
- Anomaly detection

```bash
python financial_example.py
```

### 3. Quantum Database Quickstart

**File**: `quantum_db_quickstart.py`

Comprehensive tutorial covering:
- Database initialization
- All query modes (PRECISE, BALANCED, EXPLORATORY)
- Advanced quantum features
- Performance optimization

```bash
python quantum_db_quickstart.py
```

### 4. V3.2 ML Training Examples

**File**: `src/q_store_examples/examples_v3_2.py`

Complete quantum ML training demonstrations:
- Basic quantum neural network training
- Quantum data encoding strategies
- Transfer learning with quantum models
- Multiple backend comparison
- Database-ML integration
- Quantum autoencoder

```bash
# Mock mode (no API keys needed)
python src/q_store_examples/examples_v3_2.py

# Real backends (uses .env configuration)
python src/q_store_examples/examples_v3_2.py --no-mock

# With specific credentials
python src/q_store_examples/examples_v3_2.py --no-mock \
  --pinecone-api-key YOUR_KEY \
  --ionq-api-key YOUR_KEY
```

### 5. V3.3 High-Performance ML Training

**File**: `src/q_store_examples/examples_v3_3.py`

**24-48x faster training** with algorithmic optimization:
- SPSA gradient estimation (2 circuits instead of 96)
- Hardware-efficient quantum layers (33% fewer parameters)
- Adaptive gradient optimization
- Circuit caching and batching
- Performance tracking and comparison
- Real-time speedup analysis

```bash
# Mock mode (default)
python src/q_store_examples/examples_v3_3.py

# Real backends
python src/q_store_examples/examples_v3_3.py --no-mock

# See all options
python src/q_store_examples/examples_v3_3.py --help
```

**Performance Improvements**:
- 🚀 48x fewer circuits with SPSA (2 vs 96 per batch)
- ⚡ 33% fewer parameters with hardware-efficient ansatz
- 💾 Circuit caching eliminates redundant compilations
- 🔄 Batch execution enables parallel quantum jobs

### 6. V3.4 Performance-Optimized ML Training ⚡ RECOMMENDED

**File**: `src/q_store_examples/examples_v3_4.py`

**8-10x faster than v3.3.1** through true parallelization:
- **IonQBatchClient**: Single API call for all circuits (12x faster submission)
- **IonQNativeGateCompiler**: GPi/GPi2/MS native gates (30% faster execution)
- **SmartCircuitCache**: Template-based caching (10x faster preparation)
- **CircuitBatchManagerV34**: Orchestrates all optimizations together
- Production training workflow with full v3.4 features
- Configuration guide and performance evolution analysis

```bash
# Mock mode (safe testing)
python src/q_store_examples/examples_v3_4.py

# Real backends (uses .env file)
python src/q_store_examples/examples_v3_4.py --no-mock

# With specific credentials (overrides .env)
python src/q_store_examples/examples_v3_4.py --no-mock \
  --pinecone-api-key YOUR_PINECONE_KEY \
  --ionq-api-key YOUR_IONQ_KEY \
  --ionq-target simulator
```

**Performance Targets**:
- 📊 Batch time: 35s (v3.3.1) → 4s (v3.4) = **8.75x faster**
- ⚡ Circuits/sec: 0.57 (v3.3.1) → 5.0 (v3.4) = **8.8x throughput**
- 🚀 Training time: 29.6 min (v3.3.1) → 3.75 min (v3.4) = **7.9x faster**

**What Each Example Demonstrates**:

| Example | Component | Performance Gain |
|---------|-----------|------------------|
| Example 1 | IonQBatchClient | 1 API call vs 20 |
| Example 2 | IonQNativeGateCompiler | GPi/GPi2/MS native gates |
| Example 3 | SmartCircuitCache | Template-based caching |
| Example 4 | CircuitBatchManagerV34 | All optimizations integrated |
| Example 5 | Production Training | Complete workflow with v3.4 |
| Example 6 | Configuration Guide | 4 config scenarios |
| Example 7 | Performance Evolution | v3.2 → v3.4 comparison |


### 7. ML Training Example

**File**: `ml_training_example.py`

Machine learning integration:
- Model embedding storage
- Training data selection
- Curriculum learning
- Hard negative mining

```bash
python ml_training_example.py
```

### 8. Connection Tests

Verify Pinecone and IonQ connections:

```bash
# Using .env file (recommended)
python test_pinecone_ionq_connection.py
python test_cirq_adapter_fix.py

# Or set environment variables explicitly
export PINECONE_API_KEY="your-key"
export IONQ_API_KEY="your-key"
python test_pinecone_ionq_connection.py
```

**Tests verify**:
- ✅ Pinecone client initialization and index creation
- ✅ IonQ backend configuration (simulator and QPU)
- ✅ Quantum circuit execution on IonQ
- ✅ Small training session with real backends
- ✅ Pinecone index creation during training

### 9. TinyLlama React Training

**File**: `tinyllama_react_training.py`

Complete LLM fine-tuning workflow:
- React code dataset generation
- Quantum-enhanced data sampling
- LoRA fine-tuning
- Curriculum learning

```bash
# Automated workflow
./run_react_training.sh

# Step-by-step
python react_dataset_generator.py
python tinyllama_react_training.py
```

See `REACT_QUICK_REFERENCE.md` in the examples repository for detailed instructions.


## Installation Options

### Option 1: Editable Install (Recommended)

```bash
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples
pip install -e .
```

### Option 2: Using requirements.txt

```bash
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples
pip install -r requirements.txt
pip install -e .
```

### Option 3: Using conda

```bash
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples
conda create -n q-store-examples python=3.11
conda activate q-store-examples
pip install -e .
```

### Option 4: Using Local Wheel File

If you have the Q-Store wheel file:

```bash
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples
cp /path/to/q_store-3.4.3-*.whl .
make install-wheel
# Or manually:
# pip install q_store-3.4.3-*.whl
# pip install -e .
```

### Option 5: Minimal Installation

For core functionality without ML dependencies:

```bash
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples
pip install -r requirements-minimal.txt
```

This allows running:
- `basic_example.py`
- `financial_example.py`
- `quantum_db_quickstart.py`

## Testing & Verification

### Verify Installation

```bash
# Test Q-Store installation
python verify_installation.py

# Check configuration
python show_config.py

# Test React integration
python verify_react_integration.py

# Test TinyLlama setup
python verify_tinyllama_example.py
```

### Run Unit Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov black isort flake8 mypy

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

## Usage Tips

### GPU Support

For CUDA GPU support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### Memory Management

For large datasets or limited RAM:

```python
config = TrainingConfig(
    per_device_train_batch_size=1,  # Smaller batches
    gradient_accumulation_steps=16,  # Accumulate gradients
    max_samples=500                   # Limit dataset size
)
```

### Development Mode

```bash
# Install Q-Store in editable mode
cd /path/to/q-store
pip install -e .

# Or upgrade from PyPI
pip install --upgrade q-store
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: q_store` | `pip install q-store==3.4.3` |
| `PINECONE_API_KEY not found` | Create `.env` file with your API key |
| `ImportError: transformers` | `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce batch size or use CPU |
| Dataset file not found | Run dataset generator first |
| Pinecone index not created | Ensure API key is valid, check `--no-mock` flag |

### Debug Mode

Enable verbose logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
echo "LOG_LEVEL=DEBUG" >> .env
```

## Documentation

The examples repository includes comprehensive documentation:

| File | Description |
|------|-------------|
| `REACT_QUICK_REFERENCE.md` | Quick start for React training |
| `REACT_TRAINING_WORKFLOW.md` | Detailed React training guide |
| `TINYLLAMA_TRAINING_README.md` | TinyLlama fine-tuning guide |
| `IMPROVEMENTS_SUMMARY.md` | Code improvements and comparisons |
| `SETUP.md` | Detailed setup instructions and troubleshooting |

## Related Resources

- [Q-Store Main Repository](https://github.com/yucelz/q-store)
- [Q-Store Documentation](https://q-store-docs.example.com)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [IonQ Documentation](https://ionq.com/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## Support

- **GitHub Repository**: [yucelz/q-store-examples](https://github.com/yucelz/q-store-examples)
- **Issues**: [GitHub Issues](https://github.com/yucelz/q-store/issues)
- **Configuration Help**: Run `python show_config.py` in the examples directory

---

Ready to explore quantum-enhanced machine learning? Start with v3.4 for best performance! 🚀
