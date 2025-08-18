# Plots Directory

This directory contains visualization tools and generated figures for analyzing the performance of the blockchain-enabled federated learning Digital Twin system. It provides comprehensive plotting utilities and visual results from the multi-variate time-series forecasting experiments.

## Overview

The plots directory serves as the visualization hub for the federated learning system, offering tools to analyze training performance, model convergence, and prediction accuracy across different sensor types and federated learning nodes.

## Directory Structure

### Core Plotting Scripts

#### `plot_losses.py`
**Training loss visualization tool** for federated learning analysis:

##### **Key Features:**
- **Multi-node Loss Tracking**: Visualize training losses across all 8 federated nodes
- **Sensor-specific Analysis**: Separate loss curves for temperature, humidity, and CO2
- **Convergence Analysis**: Track model convergence over 30 training rounds
- **Comparative Visualization**: Compare local vs. global model performance

##### **Generated Visualizations:**
```python
# Individual node performance
plot_node_losses(node_id, sensor_type, rounds)

# Cross-node comparison
plot_comparative_losses(all_nodes, sensor_type)

# Global convergence tracking
plot_global_convergence(aggregated_losses)

# Training round progression
plot_round_progression(loss_history)
```

##### **Output Formats:**
- Line plots showing loss reduction over training rounds
- Box plots for loss distribution across nodes
- Heatmaps for node-sensor performance matrix
- Convergence rate analysis charts

#### `plot_predictions.py`
**Prediction accuracy and forecasting visualization tool**:

##### **Functionality:**
- **Time-series Forecasting Plots**: Actual vs. predicted sensor values
- **Multi-step Prediction**: 12-hour ahead forecasting visualization
- **Uncertainty Quantification**: Confidence intervals for predictions
- **Seasonal Pattern Analysis**: Daily and weekly patterns in sensor data

##### **Visualization Types:**
```python
# Prediction accuracy plots
plot_prediction_accuracy(predictions, actuals, sensor_type)

# Time-series forecasting
plot_forecast_comparison(historical, predicted, confidence_intervals)

# Multi-variate prediction
plot_multivariate_forecast(temp_pred, humidity_pred, co2_pred)

# Error distribution analysis
plot_prediction_errors(errors_by_sensor, errors_by_room)
```

### Generated Figures Directory

#### `figures/`
**Repository of generated visualization outputs** from the federated learning experiments:

##### **Loss Analysis Figures:**

###### `hum_line.png`
- **Content**: Humidity sensor training loss progression across all federated nodes
- **X-axis**: Training rounds (1-30)
- **Y-axis**: Mean Squared Error (MSE) loss
- **Lines**: Individual curves for each of the 8 federated nodes
- **Insight**: Shows clear improvement in humidity prediction accuracy as training progresses

###### `co2_line.png`
- **Content**: CO2 sensor training loss progression for all participating nodes
- **Analysis**: Demonstrates convergence patterns for CO2 level prediction
- **Performance**: Most rooms show steady loss reduction indicating effective learning
- **Federated Benefit**: Illustrates how collaborative training improves individual node performance

###### `temp_line.png`
- **Content**: Temperature sensor training loss across federated learning rounds
- **Comparison**: Temperature predictions show different convergence rates compared to other sensors
- **Room Variations**: Different rooms exhibit varying prediction difficulty for temperature

##### **Additional Analysis Figures:**
- **Convergence Comparison**: Side-by-side sensor performance analysis
- **Node Performance Matrix**: Heatmap showing best/worst performing nodes per sensor
- **Training Dynamics**: Animation-style plots showing training progression
- **Cross-validation Results**: Performance on held-out test data

#### `lightning_logs/`
**PyTorch Lightning logging directory** containing:
- **TensorBoard Logs**: Interactive training visualizations
- **Experiment Tracking**: Multiple training run comparisons
- **Hyperparameter Logs**: Parameter sensitivity analysis
- **Model Architecture Logs**: Network structure visualizations
