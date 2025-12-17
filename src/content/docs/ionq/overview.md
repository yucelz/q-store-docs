---
title: IonQ Overview
description: Overview of IonQ quantum computing integration in Q-Store
---

Q-Store uses **IonQ** trapped-ion quantum computers as the quantum backend, providing high-fidelity quantum operations for database features.

## Why IonQ?

### Technical Advantages

**All-to-All Connectivity**
- Any qubit can interact with any other
- No SWAP gates needed
- Reduces circuit depth significantly

**High Fidelity**
- Single-qubit gates: >99.5% fidelity
- Two-qubit gates: >97% fidelity
- Industry-leading error rates

**Native Gate Set**
- Single-qubit: RX, RY, RZ, arbitrary rotations
- Two-qubit: XX gate (Mølmer-Sørensen)
- Efficient for our encoding schemes

**Scalability Roadmap**
- Current: 36 qubits (Forte)
- 2026: 64+ qubits
- 2028: 100+ qubits with error correction

## Next Steps

- Learn about [SDK Integration](/ionq/sdk-integration)
- Understand [Hardware Selection](/ionq/hardware-selection)
- See [IonQ Optimizations](/ionq/optimizations)
- Check [Production Deployment](/deployment/cloud)
