---
title: Entanglement Registry
description: Component managing entangled quantum states and correlations
---

The **Entanglement Registry** tracks and manages entangled groups of related entities, enabling automatic correlation updates.

## Responsibilities

- Track entangled groups of related entities
- Create entangled quantum states (GHZ, Bell pairs)
- Propagate updates via quantum correlation
- Measure entanglement strength (correlation)

## Key Methods

### create_entangled_group()

Creates an entangled group of related entities.

```python
create_entangled_group(
    group_id: str,
    entity_ids: List[str],
    correlation_strength: float = 0.8
) -> EntanglementGroup
```

**Example:**

```python
# Create entangled portfolio
registry.create_entangled_group(
    group_id='tech_stocks',
    entity_ids=['AAPL', 'MSFT', 'GOOGL'],
    correlation_strength=0.85
)
```

### update_entity()

Updates an entity and propagates via entanglement.

```python
update_entity(
    entity_id: str,
    new_data: Vector
) -> None
```

**Automatic Propagation:**

```python
# Update AAPL
registry.update_entity('AAPL', new_embedding)

# MSFT and GOOGL automatically updated via quantum correlation
# No manual sync needed!
```

### get_entangled_partners()

Returns all entities entangled with given entity.

```python
get_entangled_partners(
    entity_id: str
) -> List[str]
```

### measure_correlation()

Measures correlation strength between two entities.

```python
measure_correlation(
    entity_a: str,
    entity_b: str
) -> float
```

Returns correlation value 0-1.

## Entangled States

### GHZ State

Multi-party entanglement for N entities:

```
|GHZ⟩ = 1/√2(|000...0⟩ + |111...1⟩)
```

Used for groups with uniform correlation.

### Bell Pairs

Two-party entanglement:

```
|Φ⁺⟩ = 1/√2(|00⟩ + |11⟩)
```

Used for pairwise correlations.

## Implementation

```python
class EntanglementRegistry:
    def __init__(self):
        self.groups: Dict[str, EntanglementGroup] = {}
        self.entity_to_groups: Dict[str, Set[str]] = {}
    
    def create_entangled_group(
        self,
        group_id: str,
        entity_ids: List[str],
        correlation_strength: float = 0.8
    ) -> EntanglementGroup:
        # Create quantum state
        quantum_state = self._create_ghz_state(
            entity_ids,
            correlation_strength
        )
        
        # Register group
        group = EntanglementGroup(
            id=group_id,
            entities=entity_ids,
            quantum_state=quantum_state,
            correlation=correlation_strength
        )
        
        self.groups[group_id] = group
        
        # Update entity mappings
        for entity_id in entity_ids:
            if entity_id not in self.entity_to_groups:
                self.entity_to_groups[entity_id] = set()
            self.entity_to_groups[entity_id].add(group_id)
        
        return group
    
    def _create_ghz_state(
        self,
        entity_ids: List[str],
        correlation: float
    ) -> QuantumState:
        # Build GHZ circuit
        n_entities = len(entity_ids)
        qubits = cirq.LineQubit.range(n_entities)
        circuit = cirq.Circuit()
        
        # Hadamard on first qubit
        circuit.append(cirq.H(qubits[0]))
        
        # CNOTs to entangle all
        for i in range(n_entities - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        
        # Apply correlation strength
        rotation_angle = np.arccos(correlation)
        for qubit in qubits:
            circuit.append(cirq.ry(rotation_angle)(qubit))
        
        return QuantumState(circuit=circuit, qubits=qubits)
```

## Next Steps

- See [Quantum Circuit Builder](/components/circuit-builder)
- Learn about [Tunneling Engine](/components/tunneling-engine)
- Check [Classical Backend](/components/classical-backend)
