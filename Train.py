import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time
import os
import json
import logging
import argparse
from datetime import datetime
from tqdm import trange
import copy

from models import MeshODENet


class MultiStageScheduler:
    """
    Custom multi-stage learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        stages (list): List of dicts with 'epochs', 'lr', and optional 'weight_decay'
    """
    
    def __init__(self, optimizer, stages):
        self.optimizer = optimizer
        self.stages = stages
        self.current_stage = 0
        self.epoch_in_stage = 0
        self.total_epochs = 0
        
        # Set initial learning rate
        if stages:
            self._update_lr(stages[0]['lr'], stages[0].get('weight_decay', None))
    
    def step(self):
        """Step the scheduler."""
        self.epoch_in_stage += 1
        self.total_epochs += 1
        
        current_stage_info = self.stages[self.current_stage]
        
        # Check if current stage is finished
        if self.epoch_in_stage >= current_stage_info['epochs']:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.epoch_in_stage = 0
                next_stage = self.stages[self.current_stage]
                self._update_lr(next_stage['lr'], next_stage.get('weight_decay', None))
                logging.info(f"Advanced to stage {self.current_stage + 1}, "
                           f"LR: {next_stage['lr']}, WD: {next_stage.get('weight_decay', 'unchanged')}")
    
    def _update_lr(self, lr, weight_decay=None):
        """Update learning rate and optionally weight decay."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if weight_decay is not None:
                param_group['weight_decay'] = weight_decay
    
    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_stage_info(self):
        """Get current stage information."""
        return {
            'stage': self.current_stage + 1,
            'total_stages': len(self.stages),
            'epoch_in_stage': self.epoch_in_stage + 1,
            'epochs_in_stage': self.stages[self.current_stage]['epochs'],
            'current_lr': self.get_current_lr()
        }


class TrainingLogger:
    """Training logger for comprehensive logging and history tracking."""
    
    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(log_dir, f"{model_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'stage': [],
            'time_per_epoch': []
        }
    
    def load_existing_history(self):
        """Load existing history from CSV if it exists."""
        csv_path = os.path.join(self.log_dir, f"{self.model_name}_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for key in self.history.keys():
                if key in df.columns:
                    self.history[key] = df[key].tolist()
            logging.info(f"Loaded existing history from {csv_path}")
            return True
        return False
    
    def log_epoch_data(self, epoch, train_loss, test_loss, lr, stage_info, epoch_time):
        """Record epoch data to history without INFO output."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['test_loss'].append(test_loss)
        self.history['learning_rate'].append(lr)
        self.history['stage'].append(stage_info['stage'])
        self.history['time_per_epoch'].append(epoch_time)
    
    def save_history(self, verbose=True):
        """Save training history to CSV."""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.log_dir, f"{self.model_name}_history.csv")
        df.to_csv(csv_path, index=False)
        if verbose:
            logging.info(f"Training history saved to {csv_path}")
    
    def save_final_summary(self, total_time, best_test_loss, final_model_path):
        """Save final training summary."""
        summary = {
            'model_name': self.model_name,
            'total_training_time': total_time,
            'total_epochs': len(self.history['epoch']),
            'best_test_loss': best_test_loss,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_test_loss': self.history['test_loss'][-1] if self.history['test_loss'] else None,
            'final_lr': self.history['learning_rate'][-1] if self.history['learning_rate'] else None,
            'model_path': final_model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.log_dir, f"{self.model_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Training summary saved to {summary_path}")


def split_trajectory_with_tspan_and_weights(traj, num_segments=5, min_length=50, time_step_len=0.1):
    """
    Randomly split trajectory with time spans and weights.
    
    Args:
        traj (dict): Trajectory data
        num_segments (int): Number of segments to split into
        min_length (int): Minimum length of each segment
        time_step_len (float): Time step length
        
    Returns:
        tuple: (segments, t_spans, weights, t_marks)
    """
    total_steps = len(next(iter(traj.values())))
    
    # Generate random segment lengths
    remaining_steps = total_steps - num_segments * min_length
    segment_lengths = [min_length] * num_segments
    
    for i in range(remaining_steps):
        segment_lengths[random.randint(0, num_segments - 1)] += 1
    
    # Generate split points
    split_points = [0]
    for length in segment_lengths:
        split_points.append(split_points[-1] + length)
    
    # Split trajectory
    segments = []
    t_spans = []
    t_marks = []
    for i in range(num_segments):
        start, end = split_points[i], split_points[i + 1]
        t_mark = (start, end)
        t_span = torch.linspace(start, end-1, end-start) * time_step_len
        
        segment = {key: value[start:end] for key, value in traj.items()}
        segments.append(segment)
        t_marks.append(t_mark)
        t_spans.append(t_span)
    
    # Calculate weights
    weights = [length / total_steps for length in segment_lengths]
    
    return segments, t_spans, weights, t_marks


def build_optimizer(args, params):
    """Build optimizer based on configuration."""
    filter_fn = filter(lambda p: p.requires_grad, params)
    
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    
    return optimizer


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, test_loss, checkpoint_dir, model_name):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    
    # Add scheduler state if scheduler exists
    if scheduler is not None:
        checkpoint['scheduler_state'] = {
            'current_stage': scheduler.current_stage,
            'epoch_in_stage': scheduler.epoch_in_stage,
            'total_epochs': scheduler.total_epochs
        }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        logging.info("No checkpoint found, starting from scratch")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore scheduler state if it exists
    if 'scheduler_state' in checkpoint and scheduler is not None:
        scheduler.current_stage = checkpoint['scheduler_state']['current_stage']
        scheduler.epoch_in_stage = checkpoint['scheduler_state']['epoch_in_stage']
        scheduler.total_epochs = checkpoint['scheduler_state']['total_epochs']
        
        # Update optimizer with current stage LR
        current_stage_info = scheduler.stages[scheduler.current_stage]
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_stage_info['lr']
            if 'weight_decay' in current_stage_info:
                param_group['weight_decay'] = current_stage_info['weight_decay']
    
    start_epoch = checkpoint['epoch'] + 1
    best_test_loss = checkpoint.get('test_loss', float('inf'))
    
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    logging.info(f"Resuming from epoch {start_epoch}, best test loss: {best_test_loss:.6f}")
    
    return start_epoch, best_test_loss


def test(dataset_test, device, test_model, stats_list, args):
    """Calculate test set losses."""
    total_loss = 0
    num_loops = 0
    test_model.eval()
    
    with torch.no_grad():
        for traj in dataset_test:
            traj = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                   for key, value in traj.items()}
            
            segments, t_spans, weights, t_marks = split_trajectory_with_tspan_and_weights(
                traj, 
                num_segments=args.num_segments, 
                min_length=args.min_length, 
                time_step_len=args.time_step
            )
            
            traj_loss = torch.tensor(0.0, device=device)
            for segment, t_span, weight in zip(segments, t_spans, weights):
                t_span = t_span.to(device)
                pred_pos, pred_vel = test_model(segment, stats_list, t_span)
                loss = test_model.loss(pred_pos, segment)
                weight = torch.tensor(weight).to(device)
                traj_loss = traj_loss + weight * loss
            
            total_loss += traj_loss.item()
            num_loops += 1
    
    test_model.train()
    return total_loss / num_loops


def train(dataset_train, dataset_test, device, stats_list, args):
    """Main training function with comprehensive logging and checkpointing."""
    
    # Generate model name
    model_name = (f"MeshODENet_nl{args.num_layers}_bs{args.batch_size}_"
                 f"hd{args.hidden_dim}_ep{args.epochs}_wd{args.weight_decay}_"
                 f"lr{args.lr}_seg{args.num_segments}_v{args.version}")
    
    # Initialize logger
    logger = TrainingLogger(args.log_dir, model_name)
    logging.info(f"Starting training: {model_name}")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {vars(args)}")
    
    # Move stats to device
    stats_list = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                 for key, value in stats_list.items()}
    
    # Initialize model
    num_classes = 3  # Velocity dimension
    model = MeshODENet(
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        node_size=stats_list['node_size'].item(),
        edge_size=stats_list['edge_size'].item(),
        num_layers=args.num_layers
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = build_optimizer(args, model.parameters())
    scheduler = MultiStageScheduler(optimizer, args.lr_stages)
    
    # Load checkpoint if resuming
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}_checkpoint.pt")
    start_epoch, best_test_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    # Load existing history if resuming
    if start_epoch > 0:
        logger.load_existing_history()
    
    # Training loop
    start_time = time.time()
    best_model = None
    last_best_model_epoch = None
    last_best_model_loss = None
    
    try:
        for epoch in trange(start_epoch, args.epochs, desc="Training", unit="Epochs"):
            epoch_start_time = time.time()
            
            # Training phase
            total_loss = 0
            model.train()
            num_loops = 0
            
            for traj in dataset_train:
                traj = {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                       for key, value in traj.items()}
                
                segments, t_spans, weights, t_marks = split_trajectory_with_tspan_and_weights(
                    traj, 
                    num_segments=args.num_segments, 
                    min_length=args.min_length, 
                    time_step_len=args.time_step
                )
                
                optimizer.zero_grad()
                traj_loss = torch.tensor(0.0, device=device)
                
                for segment, t_span, weight in zip(segments, t_spans, weights):
                    t_span = t_span.to(device)
                    pred_pos, pred_vel = model(segment, stats_list, t_span)
                    loss = model.loss(pred_pos, segment)
                    weight = torch.tensor(weight).to(device)
                    traj_loss = traj_loss + weight * loss
                
                traj_loss.backward()
                total_loss += traj_loss.item()
                optimizer.step()
                num_loops += 1
            
            avg_train_loss = total_loss / num_loops
            
            # Test phase
            test_loss = test(dataset_test, device, model, stats_list, args)
            
            # Update scheduler
            scheduler.step()
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
                last_best_model_epoch = epoch
                last_best_model_loss = test_loss
                # Save best model immediately
                best_model_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
                torch.save(best_model.state_dict(), best_model_path)
            
            # Record epoch data (every epoch)
            epoch_time = time.time() - epoch_start_time
            stage_info = scheduler.get_stage_info()
            logger.log_epoch_data(epoch, avg_train_loss, test_loss, 
                                scheduler.get_current_lr(), stage_info, epoch_time)
            
            # Output INFO for every log_interval epochs
            should_log = (epoch % args.log_interval == 0)
            if should_log:
                logging.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, "
                    f"Test Loss: {test_loss:.6f}, LR: {scheduler.get_current_lr():.2e}, "
                    f"Stage: {stage_info['stage']}/{stage_info['total_stages']}, "
                    f"Time: {epoch_time:.2f}s")
                logging.info(f"Checkpoint saved to {checkpoint_path}")
                
                # If best model was updated in this interval, log the last update
                if last_best_model_epoch is not None:
                    logging.info(f"Best model updated at epoch {last_best_model_epoch}: test loss {last_best_model_loss:.6f}")
                    last_best_model_epoch = None
                    last_best_model_loss = None
            
            # Save checkpoint every epoch for safety
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          avg_train_loss, test_loss, args.checkpoint_dir, model_name)
            
            # Save history every epoch for real-time monitoring
            logger.save_history(verbose=False)
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    # Save final results
    total_time = time.time() - start_time
    
    # Save final checkpoint
    final_epoch = start_epoch + len(logger.history['epoch']) - 1 if logger.history['epoch'] else start_epoch
    save_checkpoint(model, optimizer, scheduler, final_epoch, 
                   avg_train_loss, test_loss, args.checkpoint_dir, model_name)
    
    # Save training history and summary
    logger.save_history()
    best_model_path = os.path.join(args.checkpoint_dir, f"{model_name}_best.pt")
    logger.save_final_summary(total_time, best_test_loss, best_model_path)
    
    logging.info(f"Training completed in {total_time:.2f} seconds")
    logging.info(f"Best test loss: {best_test_loss:.6f}")
    
    return logger.history, best_model, best_test_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MeshODENet Training")
    
    # Model parameters
    parser.add_argument('--num_layers', type=int, default=1, help='Number of processor layers')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (not used in current implementation)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop', 'adagrad'])
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # Data parameters
    parser.add_argument('--num_segments', type=int, default=1, help='Number of trajectory segments')
    parser.add_argument('--min_length', type=int, default=0, help='Minimum segment length')
    parser.add_argument('--time_step', type=float, default=0.1, help='Time step length')
    parser.add_argument('--train_size', type=int, default=30, help='Number of training trajectories')
    parser.add_argument('--test_size', type=int, default=6, help='Number of test trajectories')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Log directory')
    parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
    
    # Other
    parser.add_argument('--version', type=str, default='v1', help='Version identifier')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--log_interval', type=int, default=10, help='Epoch interval for INFO logging')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Define multi-stage learning rate schedule
    args.lr_stages = [
        {'epochs': 200, 'lr': 1e-4, 'weight_decay': 5e-4},
        {'epochs': 200, 'lr': 1e-5, 'weight_decay': 5e-4},
        {'epochs': 200, 'lr': 1e-6, 'weight_decay': 5e-4},
    ]
    
    print(f"Using device: {device}")
    print(f"Loading data from {args.data_dir}")
    
    processed_dir = os.path.join(args.data_dir, 'processed_dataset')
    # Load data
    dataset_train = torch.load(os.path.join(processed_dir, 'train.pt'))
    dataset_test = torch.load(os.path.join(processed_dir, 'test.pt'))
    stats_list = torch.load(os.path.join(processed_dir, 'stats_train.pt'))
    
    print(f"Loaded {len(dataset_train)} training and {len(dataset_test)} test trajectories")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Start training
    history, best_model, best_test_loss = train(dataset_train, dataset_test, device, stats_list, args)
    
    print(f"Training completed. Best test loss: {best_test_loss:.6f}")


if __name__ == "__main__":
    main() 