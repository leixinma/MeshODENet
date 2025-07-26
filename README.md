# MeshODENet: Neural ODE-based Graph Neural Network for Mesh Simulation

A physics-based mesh simulation framework combining Graph Neural Networks with Neural ODEs for deformation modeling.

## Project Structure

```
MeshODENet_2D_simulation/
├── models/                     # Model components
│   ├── __init__.py            # Module initialization
│   ├── mlp.py                 # Multi-Layer Perceptron
│   ├── processor.py           # Graph message passing processor
│   ├── ode_function.py        # Neural ODE dynamics function
│   └── mesh_ode_net.py        # Main MeshODENet model
├── data_preprocessing.ipynb    # Data preprocessing notebook
├── evaluation.ipynb           # Model evaluation and analysis notebook
├── Train.py                   # Main training script
├── checkpoints/               # Model checkpoints (includes best models)
├── logs/                      # Training logs
├── evaluation_results/        # Evaluation results and visualizations
├── raw_dataset/               # Raw dataset files (.pickle)
└── processed_dataset/         # Processed dataset files (.pt)
```

## Model Architecture

The MeshODENet consists of the following components:

1. **MLP**: Multi-layer perceptron for feature transformation
2. **Processor**: Graph neural network layer using message passing
3. **ODEFunction**: Defines the dynamics function for the Neural ODE
4. **MeshODENet**: Main model integrating all components

## Usage

### 1. Installation

```bash
# Activate your conda environment
conda activate xxx
pip install -r requirements.txt
```

### 2. Data Preprocessing

run the data preprocessing notebook to convert raw data into training format:

```bash
# Run the preprocessing notebook
jupyter notebook data_preprocessing.ipynb
```

This will generate in `processed_dataset/`:
- `train.pt`: Processed training data
- `test.pt`: Processed test data
- `stats_train.pt`: Normalization statistics

### 3. Training

Run the training script with default parameters:

```bash
python Train.py
```

Or customize training parameters:

```bash
python Train.py --epochs 500 --hidden_dim 256 --num_layers 2 --lr 0.0001
```

### 4. Training Features

The training script includes:

- **Multi-stage learning rate scheduler**: Automatically adjusts learning rate across different stages
- **Checkpoint management**: Saves model state every epoch for resuming training
- **Comprehensive logging**: Detailed logs saved to files and console output
- **Training history tracking**: CSV files with loss curves and learning rates
- **Automatic best model saving**: Keeps the model with lowest test loss
- **Configurable logging interval**: Control INFO output frequency (default: every 10 epochs)
- **Real-time monitoring**: CSV logs updated every epoch for live tracking

### 5. Resuming Training

To resume from a checkpoint:

```bash
python Train.py --resume
```

### 6. Model Evaluation

After training, evaluate your model using the evaluation notebook:

```bash
jupyter notebook evaluation.ipynb
```

The evaluation notebook provides:
- Comprehensive performance metrics (RMSE, MAE, correlation)
- Trajectory visualization and analysis
- Animated trajectory evolution
- Detailed evaluation reports
- Performance summary tables

## Key Features

- **Physics-informed**: Incorporates boundary conditions and physical constraints
- **Multi-stage training**: Adaptive learning rate scheduling across training phases
- **Robust checkpointing**: Resume training from any saved checkpoint
- **Comprehensive logging**: Track all training metrics and hyperparameters
- **Memory efficient**: Optimized for large mesh simulations
- **Real-time monitoring**: Live tracking of training progress
- **Comprehensive evaluation**: Detailed analysis and visualization tools

## Model Parameters

- `--num_layers`: Number of graph processor layers (default: 1)
- `--hidden_dim`: Hidden dimension for embeddings (default: 128)
- `--epochs`: Total training epochs (default: 600)
- `--lr`: Initial learning rate (default: 0.0001)
- `--weight_decay`: Weight decay for regularization (default: 5e-4)
- `--num_segments`: Number of trajectory segments (default: 1)
- `--min_length`: Minimum segment length (default: 60)
- `--log_interval`: Epoch interval for INFO logging (default: 10)

## Output Files

Training generates several output files:

- `checkpoints/{model_name}_checkpoint.pt`: Latest checkpoint
- `checkpoints/{model_name}_best.pt`: Best model (lowest test loss)
- `logs/{model_name}.log`: Detailed training log
- `logs/{model_name}_history.csv`: Training metrics over time
- `logs/{model_name}_summary.json`: Final training summary

Evaluation generates:

- `evaluation_results/detailed_metrics_*.csv`: Per-trajectory metrics
- `evaluation_results/summary_stats_*.json`: Summary statistics
- `evaluation_results/report_table_*.csv`: Report-ready metrics table
- `evaluation_results/trajectory_analysis.png`: Trajectory visualization
- `evaluation_results/trajectory_evolution.mp4`: Animated trajectory
- `evaluation_results/evaluation_report.md`: Complete evaluation report
<section id="video-demo">
    <video src="evaluation_results/trajectory_evolution.mp4" controls autoplay loop style="max-width:100%; height:auto;"></video>
</section>

## Notes

- Raw data should be placed in `raw_dataset/` directory
- Processed data will be saved in `processed_dataset/` directory
- Training checkpoints are saved every epoch for safety
- Best models are saved immediately upon improvement
- Evaluation results include comprehensive visualizations and metrics 