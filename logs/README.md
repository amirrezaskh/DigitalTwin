# Logs Directory

This directory contains comprehensive logging outputs from all components of the blockchain-enabled federated learning Digital Twin system. It provides centralized monitoring and debugging capabilities for the distributed training infrastructure.

## Overview

The logs directory serves as the central repository for runtime information, error tracking, and performance monitoring across all system components. Each log file captures specific component behavior, enabling effective debugging, performance analysis, and system health monitoring.

## Log File Structure

### Core System Logs

#### `aggregator.txt`
**Central aggregator service logging**:
- **Port**: 8080
- **Role**: Global model coordination and federated averaging
- **Content**:
  - Model parameter collection from federated nodes
  - Federated averaging computation logs
  - Global model update and distribution events
  - Training round progression tracking
  - Blockchain interaction for global model storage

**Key Log Entries:**
```
[INFO] Aggregator started on port 8080
[INFO] Round 1: Collecting parameters from 8 nodes
[INFO] FedAvg aggregation completed for round 1
[ERROR] Node 3 failed to submit parameters for round 5
[INFO] Global model updated and stored on blockchain
```

#### `app1.txt`
**Express application server logging**:
- **Port**: 3000
- **Role**: System orchestration and blockchain coordination
- **Content**:
  - HTTP server startup and shutdown events
  - API endpoint access logging
  - Blockchain transaction submissions and confirmations
  - Training lifecycle management events
  - Inter-component communication logs

**Key Log Entries:**
```
[INFO] Express server started on port 3000
[INFO] Blockchain connection established
[INFO] Starting training round 1 across all nodes
[INFO] API /start/ called - initiating federated learning
[ERROR] Blockchain transaction failed - retrying
```

### Federated Node Logs

#### `node_0.txt` through `node_7.txt`
**Individual federated learning node logging**:
- **Ports**: 8000-8007 (respectively)
- **Role**: Local model training and parameter sharing
- **Content**:
  - Local data loading and preprocessing logs
  - Model training progress and metrics
  - Parameter extraction and serialization
  - Blockchain submission confirmations
  - HTTP server status and API calls

**Log Structure per Node:**
```
[INFO] Node 0 started on port 8000
[INFO] Loading data for room 227
[INFO] Local dataset size: 8760 samples
[INFO] Starting local training epoch 1/20
[INFO] Training loss: 0.245, Validation loss: 0.298
[INFO] Model parameters extracted and ready for submission
[INFO] Parameters submitted to blockchain successfully
```

#### **Node-Specific Content:**
- **node_0.txt**: Room 227 data processing and training
- **node_1.txt**: Room 212 federated learning activities
- **node_2.txt**: Room 220 sensor data training
- **node_3.txt**: Room 130 model training progress
- **node_4.txt**: Room 129 federated participation
- **node_5.txt**: Room 223 local training metrics
- **node_6.txt**: Room 313 training coordination
- **node_7.txt**: Room 312 model performance tracking