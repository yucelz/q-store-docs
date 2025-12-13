---
title: Monitoring
description: Observability and monitoring for Q-Store deployments
---

Comprehensive monitoring for production Q-Store deployments.

## Metrics to Track

### Performance Metrics
- `query_latency_p50`, `p95`, `p99`
- `quantum_circuit_execution_time`
- `classical_filter_time`
- `cache_hit_rate`

### Resource Metrics
- `active_quantum_states`
- `coherence_violations`
- `entanglement_group_size`
- `memory_usage`

### Business Metrics
- `queries_per_second`
- `quantum_speedup_ratio`
- `error_rate`
- `cost_per_query`

## Logging Strategy

```python
logger.info("quantum_query", {
    "query_id": uuid,
    "context": context,
    "mode": mode,
    "candidates": count,
    "quantum_time_ms": duration,
    "cache_hit": bool
})
```

## Integration

- Prometheus for metrics
- Grafana for dashboards
- CloudWatch/Stackdriver for logs

## Next Steps

Check [Deployment Guides](/deployment/cloud)
