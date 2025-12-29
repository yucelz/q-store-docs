---
title: Recommendation Systems
description: Using Q-Store v4.0.0 for personalized recommendations
---

Q-Store v4.0.0 enables intelligent recommendation systems through quantum entanglement, superposition-based user modeling, and context-aware retrieval.

## Core Use Cases

### 1. Multi-Context User Preferences

Use **superposition** to store user preferences across multiple contexts:

```python
from q_store import QuantumDatabase, DatabaseConfig

config = DatabaseConfig(
    enable_quantum=True,
    enable_superposition=True,
    pinecone_api_key="your-key"
)

rec_db = QuantumDatabase(config)

# Store user in superposition of behavior contexts
rec_db.insert(
    id='user_123',
    vector=user_embedding,
    contexts=[
        ('browsing', 0.5),
        ('purchasing', 0.4),
        ('reviewing', 0.1)
    ]
)

# Query collapses to purchase intent
recommendations = rec_db.query(
    vector=query_context,
    context='purchasing',
    top_k=10
)
```

**Benefits:**
- One profile serves multiple contexts
- Context-aware recommendations
- No separate models per context

### 2. Item Similarity via Entanglement

Use **entanglement** to maintain item relationships:

```python
# Create entangled group of similar movies
rec_db.create_entangled_group(
    group_id='action_movies',
    entity_ids=['movie_1', 'movie_2', 'movie_3'],
    correlation_strength=0.85
)

# Update one movie - similar movies automatically adjust
rec_db.update('movie_1', new_embedding)
# movie_2 and movie_3 automatically reflect similarity
```

**Benefits:**
- Automatic similarity updates
- Zero-latency correlation propagation
- No manual re-indexing

### 3. Diverse Recommendations via Tunneling

Use **tunneling** to escape filter bubbles:

```python
# Classical: Returns similar items (filter bubble)
classical_recs = rec_db.query(
    vector=user_profile,
    top_k=10,
    enable_tunneling=False
)

# Quantum: Discovers diverse recommendations
diverse_recs = rec_db.tunnel_search(
    query=user_profile,
    barrier_threshold=0.7,
    tunneling_strength=0.5,
    top_k=10
)
```

**Benefits:**
- Escape filter bubbles
- Discover serendipitous recommendations
- Increase user engagement

### 4. Adaptive Session Memory

Use **decoherence** for session-based recommendations:

```python
# Recent session interactions - long coherence
rec_db.insert(
    id='session_item_1',
    vector=item_embedding,
    coherence_time=3600000  # 1 hour
)

# Older session data - natural decay
rec_db.insert(
    id='session_item_2',
    vector=old_item,
    coherence_time=600000  # 10 minutes
)

# Automatic cleanup
rec_db.apply_decoherence()
```

**Benefits:**
- Recent interactions matter more
- Old data fades naturally
- No manual session management

## Complete Example: E-Commerce Recommendations

```python
from q_store import QuantumDatabase, DatabaseConfig
import numpy as np

# Initialize recommendation database
config = DatabaseConfig(
    enable_quantum=True,
    enable_superposition=True,
    enable_tunneling=True,
    pinecone_api_key=PINECONE_KEY,
    quantum_sdk='ionq',
    ionq_api_key=IONQ_KEY
)

rec_db = QuantumDatabase(config)

# 1. Store products with entanglement
electronics = {
    'laptop_1': laptop_embedding,
    'laptop_2': similar_laptop,
    'laptop_3': another_laptop
}

# Entangle similar products
rec_db.create_entangled_group(
    group_id='laptops',
    entity_ids=list(electronics.keys()),
    correlation_strength=0.8
)

# 2. Store user profiles with multiple contexts
rec_db.insert(
    id='user_123',
    vector=user_embedding,
    contexts=[
        ('browsing', 0.4),
        ('cart', 0.3),
        ('wishlist', 0.2),
        ('purchased', 0.1)
    ]
)

# 3. Get context-aware recommendations
# For browsing users
browsing_recs = rec_db.query(
    vector=user_embedding,
    context='browsing',
    top_k=20,
    filter={'category': 'electronics'}
)

# For high-intent users (cart/wishlist)
purchase_recs = rec_db.query(
    vector=user_embedding,
    context='cart',
    top_k=10,
    filter={'in_stock': True}
)

# 4. Discover diverse recommendations
diverse = rec_db.tunnel_search(
    query=user_embedding,
    barrier_threshold=0.7,
    tunneling_strength=0.6,
    top_k=10
)

# 5. Session-based recommendations
for item in session_clicks:
    rec_db.insert(
        id=f'session_{item_id}',
        vector=item_embedding,
        coherence_time=1800000,  # 30 minutes
        metadata={'timestamp': now()}
    )

session_recs = rec_db.query(
    vector=current_context,
    top_k=5,
    filter={'session': True}
)
```

## Performance Benefits

### Classical vs Quantum Approach

| Feature | Classical | Q-Store v4.0.0 |
|---------|-----------|----------------|
| **Multi-context modeling** | Separate models | Single superposition state |
| **Item similarity** | Manual updates | Auto-sync via entanglement |
| **Diversity** | Hard-coded rules | Quantum tunneling |
| **Session memory** | Manual TTL | Physics-based decoherence |
| **Similarity search** | O(N) | O(√N) |

## Best Practices

### 1. Choose Context Weights Carefully

```python
# User behavior distribution
contexts = [
    ('browse', 0.5),      # Most common
    ('add_to_cart', 0.3), # Moderate intent
    ('purchase', 0.15),   # High intent
    ('return', 0.05)      # Negative signal
]

rec_db.insert(id=user_id, vector=embedding, contexts=contexts)
```

### 2. Set Entanglement Strength

- **0.9+**: Identical/duplicate items
- **0.75-0.9**: Same category/subcategory
- **0.6-0.75**: Related items
- **< 0.6**: Use classical similarity

### 3. Optimize Tunneling Strength

- **0.2-0.4**: Conservative (stay close to profile)
- **0.5-0.7**: Balanced (recommended for discovery)
- **0.8+**: Aggressive (maximum serendipity)

### 4. Configure Coherence Times

- **Active session**: 1800000ms (30 minutes)
- **Recent history**: 86400000ms (24 hours)
- **Long-term profile**: Persistent (no decoherence)

## Use Case Patterns

### Cold Start Problem

```python
# New user with limited data
new_user_recs = rec_db.tunnel_search(
    query=minimal_user_data,
    barrier_threshold=0.8,  # High barrier
    tunneling_strength=0.7,  # Aggressive exploration
    top_k=20
)
# Finds diverse popular items beyond immediate neighbors
```

### Collaborative Filtering

```python
# Entangle similar users
rec_db.create_entangled_group(
    group_id='similar_users',
    entity_ids=['user_1', 'user_2', 'user_3'],
    correlation_strength=0.75
)

# Update one user - similar users adjust
rec_db.update('user_1', new_preferences)
```

### Content-Based + Collaborative Hybrid

```python
# Store items with multi-context
rec_db.insert(
    id='item_123',
    vector=item_embedding,
    contexts=[
        ('content_features', 0.5),
        ('user_interactions', 0.5)
    ]
)

# Query combines both signals
hybrid_recs = rec_db.query(
    vector=query_vector,
    context='user_interactions',  # Collapse to collaborative
    top_k=10
)
```

## Limitations in v4.0.0

- **Mock accuracy**: ~10-20% (use IonQ for production)
- **IonQ accuracy**: 60-75% (NISQ constraints)
- **Quantum overhead**: +30-70ms per query
- **Cost**: Both Pinecone and IonQ costs

## Debugging Recommendations

### Check Context Collapse

```python
# Verify context-specific results
browsing_recs = rec_db.query(vector=user, context='browsing')
purchase_recs = rec_db.query(vector=user, context='purchasing')

# Should return different results
overlap = len(set(browsing_recs) & set(purchase_recs)) / len(browsing_recs)
print(f"Overlap: {overlap:.1%}")  # Should be < 50%
```

### Monitor Diversity

```python
# Compare classical vs quantum diversity
classical = rec_db.query(vector=user, enable_tunneling=False)
quantum = rec_db.tunnel_search(vector=user, tunneling_strength=0.6)

# Measure diversity (e.g., category distribution)
classical_categories = [item.metadata['category'] for item in classical]
quantum_categories = [item.metadata['category'] for item in quantum]

print(f"Classical unique categories: {len(set(classical_categories))}")
print(f"Quantum unique categories: {len(set(quantum_categories))}")
```

## Next Steps

- Learn about [Financial Applications](/applications/financial) for correlation management
- Check [ML Training](/applications/ml-training) for quantum ML models
- Explore [Scientific Computing](/applications/scientific) for similarity search
- Review [Entanglement Registry](/components/entanglement-registry) for relationship management
