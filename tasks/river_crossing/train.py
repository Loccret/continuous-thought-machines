import argparse
import math
import multiprocessing # Used for GIF generation
import os
import random # Used for saving/loading RNG state

import matplotlib as mpl
mpl.use('Agg') # Use Agg backend before pyplot import
import matplotlib.pyplot as plt # Used for loss/accuracy plots
import numpy as np
np.seterr(divide='ignore', invalid='warn') # Keep basic numpy settings
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torchvision # For disabling warning
from tqdm.auto import tqdm # Used for progress bars

from autoclip.torch import QuantileClip # Used for gradient clipping
from data.custom_datasets import RiverCrossingDataset
from tasks.image_classification.plotting import plot_neural_dynamics
from models.utils import reshape_predictions, get_latest_checkpoint
from tasks.river_crossing.plotting import make_river_crossing_gif
from tasks.river_crossing.utils import prepare_model, reshape_attention_weights, reshape_inputs
from utils.housekeeping import set_seed, zip_python_code
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on River Crossing Task")

    # Model Architecture 
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='The type of model to train.')
    parser.add_argument('--river_pairs', type=int, default=3, help='Number of pairs for River Crossing puzzle.')
    parser.add_argument('--d_model', type=int, default=2048, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=512, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=8, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--heads', type=int, default=16, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=64, help='Number of neurons for output sync.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons for action sync.')
    parser.add_argument('--neuron_select_type', type=str, default='first-last', choices=['first-last', 'random', 'random-pairing'], help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=32, help='Hidden dimensions for memory MLPs.')
    parser.add_argument('--backbone_type', type=str, default='parity_backbone', choices=['parity_backbone', 'shallow-wide'] + [f'resnet{d}-{i}' for d in [18, 34, 50, 101, 152] for i in range(1, 5)], help='Type of backbone to use.')
    parser.add_argument('--positional_embedding_type', type=str, default='none', choices=['none', 'learnable-fourier', 'multi-learnable-fourier', 'custom-rotational', 'custom-rotational-1d'], help='Type of positional embedding.')
    parser.add_argument('--out_dims', type=int, nargs='+', default=None, help='Output dimensions for each layer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Use layer normalisation.')

    # Training Configuration
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--grad_clip_quantile', type=float, default=-1, help='Quantile for gradient clipping (-1 to disable).')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'multistep', 'none'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40], help='Milestones for multistep scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for multistep scheduler.')

    # Data Configuration
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True, help='Pin memory for data loading.')

    # Logging and Checkpointing
    parser.add_argument('--log_dir', type=str, default='logs/river_crossing', help='Directory to save logs and checkpoints.')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs.')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs.')
    parser.add_argument('--log_every', type=int, default=100, help='Log every N batches.')
    parser.add_argument('--save_gif_every', type=int, default=10, help='Save visualization GIF every N epochs.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')

    # Device Configuration
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training.')

    return parser.parse_args()


def river_crossing_loss(predictions, targets):
    """Compute loss for River Crossing task."""
    # predictions: [batch_size, river_pairs+1, iterations]
    # targets: [batch_size, river_pairs+1]
    # Use the final time step prediction
    final_predictions = predictions[:, :, -1]  # [batch_size, river_pairs+1]
    return torch.nn.functional.mse_loss(final_predictions, targets.float())


def river_crossing_accuracy(predictions, targets):
    """Compute accuracy for River Crossing task."""
    # predictions: [batch_size, river_pairs+1, iterations]
    # targets: [batch_size, river_pairs+1]
    # Use the final time step prediction
    final_predictions = predictions[:, :, -1]  # [batch_size, river_pairs+1]
    
    # Convert to integer predictions
    pred_states = torch.round(final_predictions).long()
    target_states = targets.long()
    
    # Check if entire state matches
    correct = (pred_states == target_states).all(dim=1)
    return correct.float().mean()


def train_epoch(model, dataloader, optimizer, device, args):
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reshape inputs for model
        inputs = reshape_inputs(inputs, args)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            predictions_raw, certainties, synchronisation = outputs
        else:
            predictions_raw = outputs
        # Reshape predictions: (B, river_pairs+1, Ticks)
        predictions = predictions_raw.reshape(predictions_raw.size(0), args.river_pairs + 1, predictions_raw.size(-1))
        
        loss = river_crossing_loss(predictions, targets)
        accuracy = river_crossing_accuracy(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        
        if batch_idx % args.log_every == 0:
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy.item():.4f}"
            })
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)


def evaluate(model, dataloader, device, args):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reshape inputs for model
            inputs = reshape_inputs(inputs, args)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                predictions_raw, certainties, synchronisation = outputs
            else:
                predictions_raw = outputs
            # Reshape predictions: (B, river_pairs+1, Ticks)
            predictions = predictions_raw.reshape(predictions_raw.size(0), args.river_pairs + 1, predictions_raw.size(-1))
            
            loss = river_crossing_loss(predictions, targets)
            accuracy = river_crossing_accuracy(predictions, targets)
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save args
    torch.save(args, os.path.join(args.log_dir, 'args.pt'))
    
    # Create datasets
    train_dataset = RiverCrossingDataset(args.river_pairs)
    val_dataset = RiverCrossingDataset(args.river_pairs)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Create model
    prediction_reshaper = [-1, args.river_pairs + 1]  # Problem specific
    
    # Set output dimensions for the task
    args.out_dims = args.river_pairs + 1
    
    model = prepare_model(prediction_reshaper, args, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create gradient clipper
    grad_clipper = QuantileClip(quantile=args.grad_clip_quantile)
    
    # Create scheduler
    if args.scheduler_type == 'cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.num_epochs,
            warmup_start_lr=1e-20,
            eta_min=1e-7
        )
    elif args.scheduler_type == 'multistep':
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.gamma,
            warmup_epochs=args.warmup_epochs
        )
    else:
        scheduler = None
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        train_accuracies = checkpoint.get('train_accuracies', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        if scheduler:
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, val_loader, device, args)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'args': args
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, os.path.join(args.log_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final checkpoint
    final_checkpoint = {
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'args': args
    }
    if scheduler:
        final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(final_checkpoint, os.path.join(args.log_dir, 'final_checkpoint.pt'))
    
    print("Training completed!")


if __name__ == "__main__":
    main()