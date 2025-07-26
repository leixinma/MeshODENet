
# MeshODENet Evaluation Report
Generated: 2025-07-26 11:18:15

## Model Configuration
- Model Path: MeshODENet_nl1_bs1_hd128_ep600_wd0.0005_lr0.0001_seg1_vv1_best.pt
- Hidden Dimension: 128
- Number of Layers: 1
- Total Parameters: 249,731
- Time Step Length: 0.1

## Training Configuration  
- Learning Rate: 0.0001
- Weight Decay: 0.0005
- Optimizer: adam
- Time Step: 0.1

## Data Configuration
- Test Data Size: 5
- Train Data Size: 30
- Evaluated Trajectories: 5
- Average Trajectory Length: 400.0 steps

## Evaluation Results
- Position RMSE: 0.000814 ± 0.000054
- Position MAE: 0.000477
- Velocity RMSE: 0.000334
- Velocity Correlation: 0.7568
- Average Inference Time: 1.045s
- Total Evaluation Time: 5.23s

## Performance Analysis
- Best Trajectory RMSE: 0.000756
- Worst Trajectory RMSE: 0.000872
- RMSE Standard Deviation: 0.000054

## Files Generated
- Detailed metrics: detailed_metrics_*.csv
- Summary statistics: summary_stats_*.json
- Paper table: paper_table_*.csv
- Metrics visualization: metrics_distribution.png
- Trajectory analysis: trajectory_analysis.png
- Animation: trajectory_evolution.mp4

## Notes
- Evaluation conducted on MeshODENet model
- All metrics computed in original (unnormalized) units
- Animations subsampled for performance
- Version: v1
