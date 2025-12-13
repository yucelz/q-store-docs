---
title: Migration Guide
description: Migrating from classical to quantum-enhanced database
---

Gradual migration path from classical to quantum-enhanced Q-Store.

## Phase 1: Dual-Write

- Write to both classical and quantum stores
- Read from classical only
- Validate quantum results

## Phase 2: Shadow Queries

- Execute both classical and quantum queries
- Compare results
- Monitor performance

## Phase 3: Gradual Rollout

- Route increasing % of queries to quantum
- Monitor metrics closely
- Maintain rollback capability

## Phase 4: Full Migration

- All queries use quantum enhancement
- Classical as backup only

## Migration Script

```python
# Migrate existing data
async def migrate_to_quantum(classical_db, quantum_db):
    vectors = classical_db.get_all()
    
    for vector in vectors:
        await quantum_db.insert(
            id=vector.id,
            vector=vector.embedding,
            contexts=[('default', 1.0)]
        )
```

## Next Steps

See [Production Patterns](/production/connection-pooling)
