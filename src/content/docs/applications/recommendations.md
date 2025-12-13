---
title: Recommendation Systems
description: Using Q-Store for personalized recommendations
---

Q-Store enables intelligent recommendation systems through entanglement and context-aware queries.

## Use Cases

### User Preference Modeling

Store user preferences in superposition:

```python
rec_db.insert(
    id='user_123',
    vector=user_embedding,
    contexts=[
        ('browsing', 0.5),
        ('purchasing', 0.4),
        ('reviewing', 0.1)
    ]
)
```

### Item Similarity

Entangle similar items:

```python
rec_db.create_entangled_group(
    group_id='similar_movies',
    entity_ids=['movie_1', 'movie_2', 'movie_3']
)
```

## Next Steps

See other [Domain Applications](/applications/financial)
