---
title: Tunneling Engine
description: Quantum tunneling for pattern discovery
---

The **Tunneling Engine** uses quantum tunneling to discover hidden patterns that classical search misses.

## Responsibilities

- Discover hidden patterns via quantum tunneling
- Find globally optimal matches
- Escape local optima in optimization
- Detect rare but important signals

## Key Methods

### tunnel_search()

Search with quantum tunneling enabled.

```python
tunnel_search(
    query: Vector,
    barrier_threshold: float
) -> List[Vector]
```

### discover_regimes()

Discover market/data regimes.

```python
discover_regimes(
    historical_data: Dataset
) -> List[Pattern]
```

### find_precursors()

Find precursor patterns for events.

```python
find_precursors(
    target_event: Event
) -> List[State]
```

## Next Steps

See other [System Components](/components/state-manager)
