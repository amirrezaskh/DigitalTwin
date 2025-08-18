# Model Transfer Module

This directory contains the blockchain-based model parameter transfer system for federated learning. It implements secure model parameter storage, retrieval, and aggregation using Hyperledger Fabric smart contracts.

## Overview

The Model Transfer module enables secure and verifiable exchange of machine learning model parameters between federated learning nodes through a permissioned blockchain network. It ensures data integrity, auditability, and consensus-based aggregation of model updates.

## Architecture

```
Federated Nodes → Model Parameters → Blockchain Chaincode → Aggregated Model
                                           ↓
                                   Distributed Ledger Storage
```

## Directory Structure

### `model-transfer-application/`
Client-side application for interacting with the model transfer blockchain network.

#### Key Components:
- **Fabric Gateway Integration**: Client SDK for blockchain communication
- **Parameter Serialization**: Convert model weights to blockchain-compatible formats
- **Transaction Management**: Submit and query model parameter transactions
- **Identity Management**: Handle cryptographic identities for secure access

#### Features:
- **Model Submission**: Upload local model parameters to blockchain
- **Parameter Retrieval**: Download aggregated global model from ledger
- **Version Control**: Track model versions and training rounds
- **Access Control**: Ensure only authorized nodes can participate

### `model-transfer-chaincode/`
Smart contract (chaincode) implementation for Hyperledger Fabric that manages model parameters on the blockchain.

#### Core Functions:

##### **Model Parameter Management**
```go
// Store model parameters from federated nodes
func (s *SmartContract) SubmitModelParameters(ctx contractapi.TransactionContextInterface, 
    nodeID string, parameters string, round int) error

// Retrieve model parameters for aggregation
func (s *SmartContract) GetModelParameters(ctx contractapi.TransactionContextInterface, 
    nodeID string, round int) (*ModelParameters, error)

// Store aggregated global model
func (s *SmartContract) StoreGlobalModel(ctx contractapi.TransactionContextInterface, 
    parameters string, round int) error
```

##### **Training Round Management**
```go
// Initialize new training round
func (s *SmartContract) InitializeRound(ctx contractapi.TransactionContextInterface, 
    round int) error

// Check if all nodes have submitted for current round
func (s *SmartContract) CheckRoundCompletion(ctx contractapi.TransactionContextInterface, 
    round int) (bool, error)

// Advance to next training round
func (s *SmartContract) AdvanceRound(ctx contractapi.TransactionContextInterface) error
```

##### **Aggregation Operations**
```go
// Perform federated averaging of model parameters
func (s *SmartContract) AggregateModelParameters(ctx contractapi.TransactionContextInterface, 
    round int) (*AggregatedModel, error)

// Validate parameter integrity before aggregation
func (s *SmartContract) ValidateParameters(ctx contractapi.TransactionContextInterface, 
    parameters string) (bool, error)
```

