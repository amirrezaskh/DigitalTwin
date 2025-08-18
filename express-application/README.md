# Express Application

This directory contains the Node.js Express application that serves as the central orchestrator and web interface for the blockchain-enabled federated learning Digital Twin system.

## Overview

The Express application acts as the middleware layer connecting the blockchain network, federated learning nodes, and user interface. It manages the training lifecycle, coordinates communication between components, and provides API endpoints for system control and monitoring.

## Architecture Role

```
    Express App ↔ Blockchain Network
        ↕
Federated Learning Nodes
        ↕
    Aggregator
```

## Files

### `app1.js`
Main Express application server that provides:

#### **Core Functionality**
- **Port**: 3000
- **Role**: Central coordinator for the entire federated learning system
- **Integration**: Connects blockchain operations with federated learning lifecycle

#### **Key Features**

##### Training Coordination
- Manages 30 training rounds across 8 federated nodes
- Coordinates timing between local training and global aggregation
- Handles training start/stop commands through API endpoints

##### Blockchain Integration
- Utilizes Hyperledger Fabric Gateway SDK for blockchain communication
- Manages model parameter storage and retrieval from distributed ledger
- Handles smart contract invocations for model management
- Implements secure identity and signing for blockchain transactions

##### API Endpoints
```javascript
GET  /api/start/     - Start federated learning training process
GET  /exit/          - Graceful shutdown of Express application
POST /api/models/    - Handle model parameter submissions
GET  /api/status/    - Check system status and training progress
```

##### Communication Hub
- HTTP client for communicating with federated learning nodes (ports 8000-8007)
- Interface with aggregator service (port 8080)
- Request routing and response handling across distributed components

#### **Blockchain Configuration**
- **MSP ID**: Org1MSP (Organization 1 Membership Service Provider)
- **Network**: Hyperledger Fabric permissioned network
- **Channel**: mychannel (default Fabric test network channel)
- **Chaincode**: model-transfer chaincode for parameter management

#### **Process Management**
- Monitors training round progression (currentRound tracking)
- Manages system state across multiple federated learning components
- Handles graceful startup and shutdown sequences

## Operational Flow

### 1. System Initialization
```javascript
// Express app startup sequence:
1. Initialize Fabric Gateway connection
2. Load cryptographic materials (certificates, private keys)
3. Establish blockchain network connection
4. Start HTTP server on port 3000
5. Register API routes and middleware
```

### 2. Training Lifecycle Management
```javascript
// Training coordination:
1. Receive start command via /api/start/
2. Signal all federated nodes to begin local training
3. Monitor training progress across nodes
4. Coordinate with aggregator for parameter collection
5. Manage blockchain storage of aggregated parameters
6. Distribute updated global model to nodes
7. Repeat for 30 training rounds
```

### 3. Blockchain Operations
```javascript
// Model parameter management:
1. Collect local model parameters from nodes
2. Submit parameters to model-transfer chaincode
3. Invoke aggregation smart contract functions
4. Retrieve aggregated global model from ledger
5. Distribute updated model to all nodes
```

## API Documentation

### Training Control
```http
GET /api/start/
Description: Initiates the federated learning training process
Response: JSON confirmation of training start
```

### System Management
```http
GET /exit/
Description: Gracefully shuts down the Express application
Response: Process termination
```

### Status Monitoring
```http
GET /api/status/
Description: Returns current system status and training progress
Response: JSON object with system state information
```

## Development and Deployment

### Local Development
```bash
cd express-application/
npm install
node app1.js
```