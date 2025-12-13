---
title: ML Model Training
description: Using Q-Store for machine learning applications
---

Q-Store enhances ML model training through quantum-accelerated data selection and hyperparameter optimization.

## Use Cases

### Training Data Selection

Use superposition to maintain multiple training contexts:

```python
ml_db.insert(
    id='example_1',
    vector=embedding,
    contexts=[
        ('classification', 0.6),
        ('regression', 0.3),
        ('clustering', 0.1)
    ]
)
```

### Hyperparameter Optimization

Use tunneling to escape local optima:

```python
best_params = ml_db.tunnel_search(
    query=current_model_state,
    barrier_threshold=0.8,
    enable_tunneling=True
)
```

## Next Steps

See other [Domain Applications](/applications/financial)
