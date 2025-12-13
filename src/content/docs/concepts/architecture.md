---
title: Architecture Overview
description: High-level architecture of the Q-Store quantum database system
---

## High-Level Design

Q-Store uses a **hybrid architecture** that combines classical and quantum components to maximize the advantages of both:

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│  (Finance, ML Training, Recommendations, Scientific)         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Quantum Database API                            │
│  • Query Interface  • Measurement Control  • Entanglement    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼──────────┐
│  Classical Store │    │  Quantum Processor │
│  (Bulk Storage)  │    │   (Active Memory)  │
│                  │    │                    │
│  • Pinecone      │◄──►│  • IonQ Quantum    │
│  • pgvector      │    │  • State Manager   │
│  • Qdrant        │    │  • Circuit Builder │
└──────────────────┘    └───────────────────┘
```

## Component Responsibilities

### Classical Component

**Purpose**: Bulk storage and coarse filtering

- Stores millions/billions of vectors
- Handles cold data efficiently
- Provides coarse filtering (top-1000 candidates)
- Mature, reliable, cost-effective
- Options: Pinecone, pgvector, Qdrant, Redis Cache

### Quantum Component

**Purpose**: Hot data processing with quantum advantages

- Stores hot data in quantum superposition
- Enables context-aware queries via measurement
- Maintains entangled relationships
- Provides tunneling-based pattern discovery
- Handles 100-1000 active vectors

## Why Hybrid?

Current quantum computers (NISQ era) have limitations:
- Limited qubits (10-100 available)
- Short coherence times (milliseconds)
- Costly execution

The hybrid architecture:
- Maximizes quantum advantages where they matter most
- Remains practical with current hardware
- Provides fallback to classical when needed
- Optimizes cost per query

## Data Flow

### Insert Operation

```
1. Vector → Classical Store (bulk storage)
2. If hot data → Quantum Encode
3. Apply superposition contexts
4. Register entanglements
5. Set coherence time
```

### Query Operation

```
1. Query → Classical Filter (top-K candidates)
2. Candidates → Quantum Processor
3. Apply superposition collapse (context)
4. Enable tunneling (if requested)
5. Quantum measurement
6. Return results
```

### Update Operation

```
1. Update classical store
2. If entangled → Update quantum state
3. Propagate via quantum correlation
4. No manual sync needed
```

## Layered Architecture (v2.0)

Production systems use a layered approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  • REST API  • GraphQL  • gRPC  • Client SDKs               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Database Management Layer                      │
│  • Connection Pool  • Transaction Manager  • Cache          │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼──────────┐            ┌────────▼─────────┐
│  Classical Store │            │  Quantum Engine  │
│                  │            │                  │
│  • Pinecone      │◄──sync───►│  • IonQ Backend  │
│  • pgvector      │            │  • Circuit Cache │
│  • Redis Cache   │            │  • State Manager │
└──────────────────┘            └──────────────────┘
```

## Scalability

### Horizontal Scaling

- Multiple API server instances
- Load balancer distribution
- Shared classical and quantum backends

### Vertical Scaling

- Increase classical storage capacity
- Access more powerful quantum hardware (Aria → Forte)
- Larger qubit systems as they become available

## Next Steps

- Learn about [Quantum Principles](/concepts/quantum-principles)
- Understand the [Hybrid Design](/concepts/hybrid-design)
- Explore [System Components](/components/state-manager)
