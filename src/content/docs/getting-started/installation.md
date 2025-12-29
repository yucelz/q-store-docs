---
title: Installation
description: Installation guide for Q-Store v4.0.0 quantum-native database
---

## Requirements

- **Python 3.11+** (Required)
- Minimum 4GB RAM
- No GPU required for mock mode
- Optional: IonQ API key for real quantum backend (get from [cloud.ionq.com](https://cloud.ionq.com))
- Optional: Pinecone API key for vector storage (get from [pinecone.io](https://www.pinecone.io))

## Quick Install (Mock Mode)

Start immediately with mock mode - no API keys needed:

```bash
git clone https://github.com/yucelz/q-store.git
cd q-store
pip install -e .
```

Mock mode provides instant execution for development and testing (10-20% accuracy).

## Installation Options

### Basic Installation

```bash
pip install -e .
```

### With PyTorch Support

For hybrid quantum-classical ML with PyTorch integration:

```bash
pip install -e ".[torch]"
```

### With TensorFlow Support

For TensorFlow integration:

```bash
pip install -e ".[tensorflow]"
```

### Full Development Installation

Complete toolchain with all dependencies:

```bash
pip install -e ".[dev,backends,all]"
```

## API Key Configuration (Optional)

API keys are only needed when using real quantum backends (`--no-mock` flag).

Create a `.env` file in the project root:

```env
# IonQ Quantum Backend (optional)
IONQ_API_KEY=your_ionq_api_key

# Pinecone Vector Database (optional)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
```

## Setup IonQ Access (Optional)

For real quantum hardware/simulator (60-75% accuracy):

1. Sign up at [cloud.ionq.com](https://cloud.ionq.com)
2. Generate an API key
3. Add to `.env` file:

```env
IONQ_API_KEY=your-api-key-here
```

Or configure in code:

```python
from q_store import DatabaseConfig

config = DatabaseConfig(
    quantum_sdk='ionq',
    ionq_api_key='your-api-key-here',
    quantum_target='simulator'  # or 'qpu' for real hardware
)
```

## Setup Pinecone (Optional)

For persistent vector storage:

1. Sign up at [pinecone.io](https://www.pinecone.io)
2. Create an index (dimension: 768+, metric: cosine)
3. Add to `.env` file:

```env
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1
```

Or configure in code:

```python
from q_store import DatabaseConfig

config = DatabaseConfig(
    pinecone_api_key='your-pinecone-key',
    pinecone_environment='us-east-1',
    pinecone_index='quantum-vectors'
)
```

## Verify Installation

Test your installation:

```bash
python -c "from q_store import QuantumCircuit; print('✓ Q-Store installed')"
```

Run a basic example:

```bash
# Mock mode (no setup required)
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

**Python Version Error**

Q-Store requires Python 3.11 or higher:

```bash
python --version  # Should show 3.11+
```

Upgrade Python if needed:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11

# macOS with Homebrew
brew install python@3.11
```

**ImportError: No module named 'q_store'**

Ensure you're in the q-store directory:

```bash
cd q-store
pip install -e .
```

**IonQ Authentication Error**

Verify your API key:
1. Check `.env` file exists and has correct key
2. Verify key is active at [cloud.ionq.com](https://cloud.ionq.com)
3. Ensure no extra spaces in `.env` file

**Pinecone Connection Error**

Check your configuration:
1. Verify API key and environment in `.env`
2. Ensure index exists and dimension matches (768+)
3. Check metric is set to 'cosine'

**Mock Mode Accuracy Warning**

Mock mode provides 10-20% accuracy for testing only. For production:
- Use IonQ backend for 60-75% accuracy
- Consider cost vs. performance tradeoffs

## Running Examples

### Mock Mode (Free, Instant)

```bash
# Basic quantum circuit
python examples/basic_usage.py

# PyTorch hybrid model
python examples/pytorch/fashion_mnist.py --samples 500 --epochs 2

# Database operations
python examples/database_demo.py
```

### Real Quantum Backend

Requires IonQ and Pinecone API keys in `.env`:

```bash
# Use real quantum hardware/simulator
python examples/pytorch/fashion_mnist.py --no-mock --samples 100 --epochs 2
```

## Upgrading from v3.x

If upgrading from Q-Store v3.x:

1. **Uninstall old version:**
   ```bash
   pip uninstall q-store
   ```

2. **Install v4.0.0:**
   ```bash
   git clone https://github.com/yucelz/q-store.git
   cd q-store
   pip install -e .
   ```

3. **Update imports:**
   ```python
   # Old (v3.x)
   from quantum_db import QuantumDatabase

   # New (v4.0.0)
   from q_store import QuantumDatabase, DatabaseConfig
   ```

4. **Review breaking changes:**
   - See [v4.0 Release Notes](/getting-started/version-4-0) for migration guide

## Next Steps

- [Quick Start Guide](/getting-started/quick-start) - Get started in 5 minutes
- [v4.0 Release Notes](/getting-started/version-4-0) - What's new
- [Example Projects](/getting-started/example-projects) - Complete examples
- [IonQ Integration](/ionq/overview) - Real quantum backend setup
