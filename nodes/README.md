# Nodes Directory

This directory contains the implementation of federated learning nodes and the central aggregator for the blockchain-enabled Digital Twin system. Each file represents a critical component in the distributed training infrastructure.

## Overview

The nodes directory implements a federated learning system where 8 nodes (representing different rooms in a smart building) collaboratively train a Temporal Fusion Transformer model. The system uses a client-server architecture with an aggregator coordinating the training process through blockchain integration.

## File Structure

### Core Components

#### `aggregator.py`
Central coordinator for the federated learning process:
- **Role**: Manages global model aggregation and training coordination
- **Port**: 8080
- **Responsibilities**:
  - Collects model parameters from all federated nodes
  - Performs FedAvg (Federated Averaging) aggregation
  - Maintains global model state
  - Coordinates training rounds through blockchain
  - Tracks global training metrics and convergence

#### `model.py`
Global model initialization and management:
- Creates the initial Temporal Fusion Transformer architecture
- Defines model hyperparameters and configuration
- Provides model serialization/deserialization utilities
- Establishes the baseline model for federated training

#### `global_var.py`
Shared configuration and utilities:
- Common imports and dependencies
- Global constants and hyperparameters
- Utility functions used across all nodes
- Data preprocessing pipelines
- Model architecture definitions

### Individual Node Implementations

#### `node0.py` through `node7.py`
Eight federated learning nodes, each representing a different room:

**Node Configuration:**
- **Ports**: 8000-8007 (node0 through node7)
- **Data**: Each node trains on room-specific sensor data
- **Local Training**: Independent model training on private datasets
- **Communication**: HTTP-based communication with aggregator and blockchain

**Node Responsibilities:**
- Load and preprocess local room sensor data
- Train local Temporal Fusion Transformer model
- Extract and share model parameters (not raw data)
- Participate in federated averaging rounds
- Maintain local model checkpoints and metrics

### Data Structure

#### `data/`
Contains room-specific sensor datasets:
- **Format**: CSV files with time-series sensor readings
- **Sensors**: Temperature, humidity, CO2 levels
- **Temporal Resolution**: Hourly measurements
- **Coverage**: Multiple rooms across the smart building
- **Privacy**: Data never leaves the local node during training

#### `models/`
Model storage and checkpointing:
- Local model checkpoints for each node
- Global model states after aggregation
- Model version tracking and rollback capabilities
- Serialized model parameters for blockchain storage

#### `losses/`
Training metrics and performance tracking:
- Per-node training loss histories
- Global model convergence metrics
- Validation performance across different sensor types
- JSON-formatted results for analysis and visualization

#### `lightning_logs/`
PyTorch Lightning logging directory:
- TensorBoard logs for training visualization
- Hyperparameter tracking
- Model performance metrics
- Training progress monitoring

## Federated Learning Process

### 1. Initialization Phase
```
model.py → Initialize global TFT model
node0.py to node7.py → Load local data and create local models
aggregator.py → Setup global coordination
```

### 2. Training Rounds
```
For each round (1 to 30):
  1. Nodes train locally on private data
  2. Nodes extract model parameters
  3. Parameters stored on blockchain
  4. Aggregator retrieves and averages parameters
  5. Updated global model distributed to nodes
```

### 3. Communication Flow
```
Express App (Port 3000) ↔ Blockchain Network
                         ↔ Aggregator (Port 8080)
                         ↔ Nodes (Ports 8000-8007)
```

## Node Architecture

### Local Training Process
1. **Data Loading**: Load room-specific CSV data
2. **Preprocessing**: Normalize and format time-series data
3. **Model Training**: Train TFT on local data (20 epochs)
4. **Parameter Extraction**: Extract model weights
5. **Blockchain Submission**: Store parameters on distributed ledger

### Privacy Preservation
- **Data Locality**: Raw data never transmitted
- **Parameter Sharing**: Only model weights are shared
- **Secure Aggregation**: Blockchain-verified parameter exchange
- **Access Control**: Permissioned network participation

## Configuration

### Model Hyperparameters
- **Encoder Length**: 96 time steps (4 days)
- **Prediction Length**: 12 time steps (12 hours)
- **Hidden Size**: 32
- **Attention Heads**: 4
- **Dropout**: 0.2
- **Learning Rate**: 3e-4
- **Batch Size**: 64

### Training Configuration
- **Total Rounds**: 30
- **Local Epochs per Round**: 20
- **Aggregation Method**: FedAvg
- **Evaluation Metrics**: RMSE for each sensor type

## Usage

Nodes are automatically started by the main orchestration script:

```bash
python3 run.py  # Starts all nodes and aggregator
```

Individual components can be tested separately:

```bash
cd nodes/
python3 aggregator.py  # Start aggregator
python3 node0.py       # Start specific node
```

## Monitoring

### Log Files
- Individual node logs: `/logs/node_X.txt`
- Aggregator logs: `/logs/aggregator.txt`
- Express application logs: `/logs/app1.txt`

### Performance Metrics
- Loss tracking in `/losses/` directory
- TensorBoard logs in `lightning_logs/`
- Blockchain transaction logs in `/test-network/log.txt`

## Dependencies

- PyTorch & PyTorch Lightning
- PyTorch Forecasting
- Pandas, NumPy, Scikit-learn
- Flask/HTTP servers for communication
- Blockchain integration libraries

The nodes directory represents the core of the federated learning implementation, enabling privacy-preserving collaborative training across multiple smart building rooms while maintaining data locality and security through blockchain integration.
