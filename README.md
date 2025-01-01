# CARLA Environment for Autonomous Driving Research

This repository contains a comprehensive environment for autonomous driving research using the CARLA simulator. It implements various deep learning models, including Bird's Eye View (BEV) perception, world forward models, and kinematic models for autonomous vehicle control and planning.

## Project Overview

This codebase focuses on developing and testing autonomous driving algorithms with the following key components:

- **World Forward Models**: Predictive models for understanding vehicle dynamics and environment evolution
- **Bird's Eye View (BEV) Processing**: Advanced perception systems using top-down view representations
- **Model Predictive Control (MPC)**: Sophisticated control algorithms for vehicle navigation
- **Dynamic and Kinematic Models**: Physics-based vehicle modeling for accurate simulation
- **Policy Training**: Various policy training implementations for autonomous driving
- **CARLA Integration**: Seamless integration with the CARLA simulator for realistic testing

## Directory Structure

- `carla_env/`: Core environment implementation for CARLA simulator
- `configs/`: Configuration files for different experiments and models
- `docs/`: Documentation and additional resources
- `figures/`: Generated figures and visualizations
- `script/`: Utility scripts for various tasks
- `utils/`: Helper functions and utility modules
- `simple_bev/`: Bird's Eye View implementation
- `leaderboard/`: Evaluation framework for autonomous driving agents
- `scenario_runner/`: Scenario definition and execution tools

## Key Features

- Dynamic Forward Model (DFM) implementation
- Kinematic Model (KM) integration
- Extended BEV perception system
- Multiple training frequencies support (5Hz, 20Hz)
- MPC-based control implementation
- Ground truth BEV model training
- Policy evaluation framework
- Data collection utilities

## Setup and Installation

1. Create a Conda environment using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate carla
   ```

3. Install CARLA simulator (compatible with version 0.9.13)

4. Set up additional dependencies:
   - PyTorch with CUDA support
   - OpenCV
   - Other required packages are listed in environment.yml

## Usage

### Basic Environment Interaction
```python
python play_carla_env.py --num_episodes 10
```

### Training Models
- World Forward Model:
  ```python
  python train_world_forward_model_ddp.py
  ```
- Policy Training:
  ```python
  python train_dfm_km_cp_extended_bev_gt_bev_encoded_policy_fused.py
  ```

### Testing and Evaluation
- MPC Testing:
  ```python
  python test_mpc_carla.py
  ```
- Policy Testing:
  ```python
  python test_policy_carla_dfm_km_cp_extended_bev_5Hz.py
  ```

## Data Collection

Use the provided scripts to collect training data:
```python
python collect_data_dynamic_kinematic_model.py
python collect_data_ground_truth_bev_model.py
```

## Model Evaluation

Evaluate trained models using:
```python
python eval_world_forward_model.py
python eval_ego_forward_model.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

Please refer to the LICENSE file for details.

## Acknowledgments

This project builds upon the CARLA simulator and various open-source autonomous driving research tools.
