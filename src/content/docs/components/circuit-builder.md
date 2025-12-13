---
title: Quantum Circuit Builder
description: Component for building and optimizing quantum circuits
---

The **Quantum Circuit Builder** generates optimized quantum circuits for IonQ hardware.

## Responsibilities

- Generate quantum circuits for IonQ hardware
- Implement amplitude encoding
- Create entanglement operations
- Build tunneling operators
- Handle measurement basis selection
- Optimize for IonQ native gates

## Key Methods

### build_encoding_circuit()

Encodes classical vector as quantum amplitudes.

```python
build_encoding_circuit(
    vector: Vector
) -> Circuit
```

### build_entanglement_circuit()

Creates entangled state for multiple vectors.

```python
build_entanglement_circuit(
    group: List[Vector]
) -> Circuit
```

### build_tunneling_circuit()

Builds circuit for quantum tunneling search.

```python
build_tunneling_circuit(
    source: Vector,
    target: Vector
) -> Circuit
```

### build_measurement_circuit()

Creates measurement in specified basis.

```python
build_measurement_circuit(
    basis: str
) -> Circuit
```

## Next Steps

See other [System Components](/components/state-manager)
