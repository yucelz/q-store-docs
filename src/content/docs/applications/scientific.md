---
title: Scientific Computing
description: Using Q-Store for scientific applications
---

Q-Store accelerates scientific computing through quantum-enhanced similarity search and pattern discovery.

## Use Cases

### Molecular Similarity

Find similar molecules with tunneling:

```python
similar = sci_db.tunnel_search(
    query=target_molecule,
    context='binding_affinity',
    barrier_threshold=0.9
)
```

### Protein Structure Comparison

Store structures with multiple contexts:

```python
sci_db.insert(
    id='protein_123',
    vector=structure_embedding,
    contexts=[
        ('folded', 0.7),
        ('unfolded', 0.2),
        ('intermediate', 0.1)
    ]
)
```

## Next Steps

See other [Domain Applications](/applications/financial)
