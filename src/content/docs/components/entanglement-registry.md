---
title: Entanglement Registry
description: Component managing entangled quantum states and feature correlations
---

The **Entanglement Registry** tracks and manages entangled groups of related entities, enabling automatic correlation updates and quantum feature learning in v3.2.

## Responsibilities

- Track entangled groups of related entities and features
- Create entangled quantum states (GHZ, Bell pairs)
- Propagate updates via quantum correlation
- Measure entanglement strength and feature correlations
- Support quantum regularization using entanglement (NEW in v3.2)
- Enable quantum data augmentation via entanglement (NEW in v3.2)

## Key Methods

### create_entangled_group()

Creates an entangled group of related entities.

**Function Signature:**
```python
create_entangled_group(
    group_id: str,
    entity_ids: List[str],
    correlation_strength: float = 0.8,
    entanglement_type: str = 'GHZ'
) -> EntanglementGroup
```

**Purpose:** Create quantum entanglement between related entities for automatic correlation propagation.

---

### create_feature_entanglement()

Creates entangled quantum features for ML (NEW in v3.2).

**Function Signature:**
```python
create_feature_entanglement(
    feature_ids: List[str],
    entanglement_pattern: str = 'linear'
) -> QuantumCircuit
```

**Purpose:** Generate entangled feature representations for quantum neural networks.

---

### update_entity()

Updates an entity and propagates via entanglement.

**Function Signature:**
```python
update_entity(
    entity_id: str,
    new_data: np.ndarray
) -> None
```

**Purpose:** Update one entity and automatically propagate changes to entangled partners via quantum correlation.

---

### get_entangled_partners()

Returns all entities entangled with given entity.

**Function Signature:**
```python
get_entangled_partners(
    entity_id: str
) -> List[str]
```

**Purpose:** Retrieve all entities in the same entanglement group.

---

### measure_correlation()

Measures correlation strength between two entities.

**Function Signature:**
```python
measure_correlation(
    entity_a: str,
    entity_b: str
) -> float
```

**Purpose:** Compute quantum correlation (returns value 0-1) between entangled entities.

---

### compute_entanglement_entropy()

Computes entanglement entropy for regularization (NEW in v3.2).

**Function Signature:**
```python
compute_entanglement_entropy(
    group_id: str
) -> float
```

**Purpose:** Calculate entanglement entropy for quantum regularization in ML training.

---

### break_entanglement()

Breaks entanglement between entities.

**Function Signature:**
```python
break_entanglement(
    group_id: str
) -> None
```

**Purpose:** Dissolve an entanglement group and remove quantum correlations.

---

### get_entangled_groups()

Returns all active entanglement groups.

**Function Signature:**
```python
get_entangled_groups() -> Dict[str, EntanglementGroup]
```

**Purpose:** Retrieve all currently tracked entanglement groups.

---

## Entangled States

### GHZ State (Multi-party Entanglement)

Multi-party entanglement for N entities:
```
|GHZ⟩ = 1/√2(|000...0⟩ + |111...1⟩)
```

**Use Cases:**
- Groups with uniform correlation
- Portfolio optimization
- Multi-feature entanglement in ML

### Bell Pairs (Two-party Entanglement)

Two-party entanglement:
```
|Φ⁺⟩ = 1/√2(|00⟩ + |11⟩)
```

**Use Cases:**
- Pairwise correlations
- Feature pair learning
- Binary relationship tracking

### Linear Entanglement (Chain Pattern)

Sequential entanglement pattern for quantum circuits:
```
Qubit 0 ←→ Qubit 1 ←→ Qubit 2 ←→ ... ←→ Qubit N
```

**Use Cases:**
- Quantum convolutional layers
- Sequential feature processing
- Scalable circuit design

### Circular Entanglement (Ring Pattern)

Circular entanglement with wrap-around:
```
Qubit 0 ←→ Qubit 1 ←→ ... ←→ Qubit N ←→ Qubit 0
```

**Use Cases:**
- Periodic boundary conditions
- Symmetric feature representations

### Full Entanglement (All-to-All)

Complete entanglement graph:
```
Every qubit pair is entangled
```

**Use Cases:**
- Maximum expressivity quantum circuits
- Dense feature interactions
- Small-scale models (limited qubits)

## ML Training Integration (v3.2)

### Quantum Regularization

Entanglement entropy can be used as a regularization term:
- Encourages quantum correlations in learned features
- Prevents overfitting to classical patterns
- Enables unique quantum advantages

### Feature Learning

Entangled features learn correlated representations:
- Automatic feature interaction discovery
- Reduced parameter space via entanglement
- Improved generalization

## Next Steps

- See [Quantum Circuit Builder](/components/circuit-builder)
- Learn about [Tunneling Engine](/components/tunneling-engine)
- Explore [ML Model Training](/applications/ml-training)
