---
title: IonQ Optimizations
description: Optimizing circuits for IonQ hardware
---

## Native Gate Set

IonQ hardware supports:
- **Single-qubit**: RX, RY, RZ
- **Two-qubit**: XX gate (Mølmer-Sørensen)

## All-to-All Connectivity

IonQ systems have full qubit connectivity - no SWAP gates needed!

## Optimization Strategies

1. **Decompose to native gates**
2. **Exploit all-to-all connectivity**
3. **Combine adjacent rotations**
4. **Minimize circuit depth**

## Next Steps

Check [Production Deployment](/deployment/cloud)
