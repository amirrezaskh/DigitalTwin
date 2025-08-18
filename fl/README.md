# Federated Learning (FL) Module

This module contains the core federated learning implementation for the Digital Twin project, focusing on training Temporal Fusion Transformer (TFT) models for multi-variate time-series forecasting.

## Overview

The FL module implements a privacy-preserving federated learning framework where multiple nodes (representing smart building rooms) collaboratively train a global model without sharing raw data. Each node trains on local data from their respective rooms and shares only model parameters through the blockchain network.

## Architecture

- **Distributed Training**: Each room acts as a federated node with private sensor data
- **Privacy Preservation**: Raw data never leaves the local node
- **Blockchain Integration**: Model parameters are stored and validated on Hyperledger Fabric
- **Temporal Fusion Transformer**: Advanced deep learning model for time-series forecasting

## Files Description

### `node.py`
Core federated learning node implementation that:
- Loads and preprocesses local sensor data (temperature, humidity, CO2)
- Trains a Temporal Fusion Transformer model locally
- Communicates with the blockchain for model parameter sharing
- Implements secure aggregation protocols

**Key Features:**
- Multi-variate time-series forecasting for 3 sensor types
- 4-day encoder length with 12-hour prediction horizon
- PyTorch Lightning-based training pipeline
- Automatic model checkpointing and loss tracking

## Model Specifications

- **Input Features**: Temperature, humidity, CO2 levels
- **Encoder Length**: 96 time steps (4 days × 24 hours)
- **Prediction Length**: 12 time steps (12 hours)
- **Architecture**: Temporal Fusion Transformer
- **Loss Function**: Multi-target RMSE
- **Batch Size**: 64
- **Learning Rate**: 3e-4

## Data Flow

1. **Local Training**: Each node trains on room-specific sensor data
2. **Parameter Extraction**: Model weights are extracted after local training
3. **Blockchain Storage**: Parameters are stored on the distributed ledger
4. **Global Aggregation**: Aggregator combines parameters from all nodes
5. **Model Update**: Updated global model is distributed back to nodes

## Privacy Features

- **Data Locality**: Raw sensor data remains on local nodes
- **Differential Privacy**: Optional noise addition for enhanced privacy
- **Secure Aggregation**: Only model parameters are shared
- **Access Control**: Blockchain-based permission management

## Usage

The FL module is typically invoked by the main orchestration system (`run.py`) and works in conjunction with:
- Node management system (`/nodes/`)
- Blockchain network (`/test-network/`)
- Express application (`/express-application/`)

## Dependencies

- PyTorch & PyTorch Lightning
- PyTorch Forecasting
- Pandas & NumPy
- Scikit-learn
- Requests (for HTTP communication)

## Performance Metrics

The module tracks:
- Local model performance (MSE loss per node)
- Training convergence rates
- Communication overhead
- Privacy preservation effectiveness

See `/plots/figures/` for visualization of training results across different sensor types and rooms.
