---
title: REST API
description: HTTP REST API reference for Q-Store
---

Q-Store provides a production-ready REST API for all database operations.

## Base URL

```
Production: https://api.q-store.io/v2
Development: http://localhost:8000/v2
```

## Authentication

All requests require an API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.q-store.io/v2/vectors
```

## Endpoints

### Insert Vector

Insert a vector with optional contexts and entanglement.

```http
POST /v2/vectors
Content-Type: application/json
```

**Request Body:**

```json
{
  "id": "doc_123",
  "vector": [0.1, 0.2, 0.3, ...],
  "contexts": [
    {"name": "technical", "weight": 0.7},
    {"name": "general", "weight": 0.3}
  ],
  "coherence_time_ms": 1000,
  "metadata": {
    "source": "arxiv",
    "date": "2025-12-13"
  }
}
```

**Response:**

```json
{
  "id": "doc_123",
  "status": "inserted",
  "quantum_encoded": true,
  "timestamp": "2025-12-13T10:30:00Z"
}
```

### Query Vectors

Search for similar vectors with quantum enhancements.

```http
POST /v2/query
Content-Type: application/json
```

**Request Body:**

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "context": "technical",
  "mode": "balanced",
  "enable_tunneling": true,
  "top_k": 10,
  "timeout_ms": 5000
}
```

**Response:**

```json
{
  "results": [
    {
      "id": "doc_456",
      "score": 0.95,
      "vector": [0.11, 0.19, 0.31, ...],
      "metadata": {"source": "arxiv"},
      "quantum_enhanced": true,
      "context_collapsed": "technical"
    }
  ],
  "query_time_ms": 245,
  "quantum_time_ms": 180,
  "cache_hit": false
}
```

### Batch Insert

Insert multiple vectors in a single transaction.

```http
POST /v2/vectors/batch
Content-Type: application/json
```

**Request Body:**

```json
{
  "vectors": [
    {
      "id": "vec1",
      "vector": [...],
      "contexts": [...]
    },
    {
      "id": "vec2",
      "vector": [...],
      "contexts": [...]
    }
  ],
  "transaction": true
}
```

**Response:**

```json
{
  "inserted": 2,
  "failed": 0,
  "transaction_id": "tx_789",
  "quantum_batch": true
}
```

### Create Entanglement

Create an entangled group of related entities.

```http
POST /v2/entanglement/groups
Content-Type: application/json
```

**Request Body:**

```json
{
  "group_id": "related_docs",
  "entity_ids": ["doc_1", "doc_2", "doc_3"],
  "correlation_strength": 0.85
}
```

**Response:**

```json
{
  "group_id": "related_docs",
  "entities": 3,
  "correlation_strength": 0.85,
  "quantum_state_created": true
}
```

### Get Entangled Partners

Retrieve entities entangled with a given entity.

```http
GET /v2/entanglement/partners/{entity_id}
```

**Response:**

```json
{
  "entity_id": "doc_1",
  "partners": [
    {
      "id": "doc_2",
      "correlation": 0.85,
      "group_id": "related_docs"
    },
    {
      "id": "doc_3",
      "correlation": 0.85,
      "group_id": "related_docs"
    }
  ]
}
```

### Update Vector

Update an existing vector (propagates via entanglement).

```http
PUT /v2/vectors/{id}
Content-Type: application/json
```

**Request Body:**

```json
{
  "vector": [0.2, 0.3, 0.4, ...],
  "metadata": {
    "updated_at": "2025-12-13T11:00:00Z"
  }
}
```

**Response:**

```json
{
  "id": "doc_123",
  "status": "updated",
  "entangled_partners_updated": 2,
  "quantum_propagation": true
}
```

### Delete Vector

Delete a vector and remove from entanglements.

```http
DELETE /v2/vectors/{id}
```

**Response:**

```json
{
  "id": "doc_123",
  "status": "deleted",
  "entanglements_removed": 1
}
```

### Apply Decoherence

Trigger cleanup of decohered states.

```http
POST /v2/maintenance/decoherence
```

**Response:**

```json
{
  "states_cleaned": 45,
  "memory_freed_mb": 12.3,
  "execution_time_ms": 89
}
```

### Health Check

Check system health.

```http
GET /v2/health
```

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "classical_backend": "healthy",
    "quantum_backend": "healthy",
    "cache": "healthy"
  },
  "timestamp": "2025-12-13T10:30:00Z"
}
```

### Metrics

Get system metrics (requires admin role).

```http
GET /v2/metrics
```

**Response:**

```json
{
  "queries_per_second": 120,
  "quantum_utilization": 0.65,
  "cache_hit_rate": 0.87,
  "avg_query_latency_ms": 245,
  "active_quantum_states": 342,
  "coherence_violations_per_hour": 12
}
```

## Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Success |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Missing/invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Quantum backend unavailable |

## Rate Limiting

```
Default: 1000 requests/minute
Burst: 100 requests/second
```

**Response Headers:**

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1702467600
```

## Pagination

For endpoints returning multiple results:

```http
GET /v2/vectors?page=2&size=50
```

**Response:**

```json
{
  "results": [...],
  "pagination": {
    "page": 2,
    "size": 50,
    "total": 10000,
    "pages": 200
  }
}
```

## Filtering

Apply filters to queries:

```http
POST /v2/query
Content-Type: application/json

{
  "vector": [...],
  "filters": {
    "metadata.source": "arxiv",
    "metadata.date": {"$gte": "2025-01-01"}
  }
}
```

## Webhooks

Configure webhooks for events:

```http
POST /v2/webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhook",
  "events": ["vector.inserted", "entanglement.created"],
  "secret": "your-webhook-secret"
}
```

## Client Examples

### Python

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "https://api.q-store.io/v2"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Insert vector
response = requests.post(
    f"{BASE_URL}/vectors",
    headers=headers,
    json={
        "id": "doc_123",
        "vector": [0.1, 0.2, 0.3],
        "contexts": [
            {"name": "technical", "weight": 0.7}
        ]
    }
)

print(response.json())
```

### JavaScript/TypeScript

```typescript
const API_KEY = "your-api-key";
const BASE_URL = "https://api.q-store.io/v2";

const response = await fetch(`${BASE_URL}/query`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    vector: [0.1, 0.2, 0.3],
    context: "technical",
    top_k: 10
  })
});

const data = await response.json();
console.log(data);
```

### cURL

```bash
curl -X POST https://api.q-store.io/v2/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "context": "technical",
    "top_k": 10
  }'
```

## Next Steps

- See [Python SDK](/api/python-sdk) for async client
- Check [Core API](/api/core) for Python library
- Review [Authentication](https://docs.q-store.io/auth) for security
