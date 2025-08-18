# Data Directory

This directory contains the sensor datasets and data processing utilities for the blockchain-enabled federated learning Digital Twin system. It manages time-series sensor data from 76 rooms in a University of Manitoba smart building.

## Overview

The Data directory serves as the central repository for:
- **Raw Sensor Data**: Time-series measurements from IoT sensors across 76 building rooms
- **Data Processing**: Jupyter notebooks for data cleaning, analysis, and preparation
- **Data Metadata**: Information about data availability and sensor types per room
- **Training Logs**: PyTorch Lightning training logs and metrics

## Directory Structure

### Core Files

#### `main.ipynb`
Primary data analysis and experimentation notebook:
- **Purpose**: Main data exploration and model development
- **Contents**: 
  - Comprehensive data analysis and visualization
  - Model architecture experimentation
  - Performance evaluation across different rooms
  - Federated learning simulation and testing
- **Usage**: Central notebook for understanding data patterns and model behavior

#### `clean.ipynb`
Data preprocessing and cleaning notebook:
- **Purpose**: Data quality assurance and preprocessing pipeline
- **Functions**:
  - Missing value detection and interpolation
  - Outlier identification and treatment
  - Data normalization and standardization
  - Time-series alignment and resampling
- **Output**: Clean, processed datasets ready for federated learning

#### `data_info.json`
Metadata catalog for all room datasets:
- **Structure**: Maps CSV files to available sensor types per room
- **Sensor Types**: 
  - `"humidity_value"`: Relative humidity measurements
  - `"co2_value"`: CO2 concentration levels  
  - `"temprature_value"`: Temperature readings (note: original spelling preserved)
- **Coverage**: Information for all 76 rooms showing sensor availability
- **Usage**: Helps determine which rooms have complete sensor suites vs. partial coverage

### Data Subdirectories

#### `DTFL - Data/`
Raw sensor data repository containing:
- **Format**: Individual CSV files per room (e.g., `227-Table 1.csv`)
- **Naming Convention**: `{room_number}-Table 1.csv`
- **Room Coverage**: 76 unique rooms across the smart building
- **Time Period**: Extended time-series data with hourly granularity
- **Sensors**: Temperature, humidity, and CO2 measurements where available

**Data Schema per CSV:**
```
- timestamp: Date/time of measurement
- room_id: Unique room identifier  
- temprature_value: Temperature in degrees (where available)
- humidity_value: Relative humidity percentage (where available)
- co2_value: CO2 concentration in ppm (where available)
- area: Room floor area
- doors_area: Total door area
- panels_area: Window/panel area
- hours: Hour of day (0-23)
- days: Day of week
- months: Month of year
```

#### `lightning_logs/`
PyTorch Lightning training logs:
- **TensorBoard Logs**: Training metrics and visualizations
- **Version Tracking**: Multiple experiment versions
- **Performance Metrics**: Loss curves, validation scores
- **Hyperparameter Logs**: Model configuration tracking

## Data Characteristics

### Sensor Coverage Analysis
Based on `data_info.json`:

**Complete Sensor Suite (Temperature + Humidity + CO2):**
- Rooms: 212, 220, 130, 129, 223, 313, 312, 412
- **Count**: 8 rooms with full sensor coverage

**CO2 Only:**
- Majority of rooms (60+ rooms)
- **Primary Sensor**: CO2 concentration monitoring

**No Sensors:**
- Rooms: 352, 123, 354, 434, 433, 353, 124, 429, 430, 227
- **Count**: 10 rooms with missing sensor data

### Data Quality Features

#### Time-Series Properties
- **Temporal Resolution**: Hourly measurements
- **Missing Value Handling**: Linear interpolation for gaps
- **Seasonality**: Daily, weekly, and monthly patterns captured
- **Stationarity**: Data preprocessing ensures model compatibility

#### Feature Engineering
- **Temporal Features**: Hour, day, month categorical encoding
- **Static Features**: Room characteristics (area, doors, panels)
- **Time Index**: Sequential time indexing for each room
- **Normalization**: Multi-target standardization for different sensor types
