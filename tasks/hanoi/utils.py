import os
import re
import math
import torch
from models.ctm import ContinuousThoughtMachine
from models.lstm import LSTMBaseline

def prepare_model(prediction_reshaper, args, device):
    if args.model_type == 'ctm':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,  
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,  
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,  
            do_layernorm_nlm=args.do_normalisation,  
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,          
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        ).to(device)
    elif args.model_type == 'lstm':
        model = LSTMBaseline(
            d_model=args.d_model,
            d_input=args.d_input,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model


def reshape_inputs(inputs, args):
    """Reshape inputs for the model."""
    # inputs shape: [batch_size, hanoi_disks]
    # For CTM/LSTM, we need: [batch_size, seq_len, input_dim]
    batch_size = inputs.size(0)
    
    # Add sequence dimension and expand input dimension
    inputs = inputs.unsqueeze(1)  # [batch_size, 1, hanoi_disks]
    
    # Pad to match expected input dimension if needed
    if inputs.size(-1) < args.d_input:
        padding = args.d_input - inputs.size(-1)
        inputs = torch.nn.functional.pad(inputs, (0, padding))
    
    return inputs


def reshape_attention_weights(attention_weights, args):
    """Reshape attention weights for visualization."""
    # attention_weights shape depends on model architecture
    # This is a placeholder - adjust based on actual model output
    return attention_weights


def get_where_most_certain(certainties, threshold=0.8):
    """Get indices where model is most certain about predictions."""
    max_certainties, _ = torch.max(certainties, dim=-1)
    return torch.where(max_certainties > threshold)[0]


def parse_folder_name(folder_name):
    """Parse folder name to extract training parameters."""
    # Example: "ctm_hanoi_D=1024_T=75_M=25_N=3"
    pattern = r"ctm_hanoi_D=(\d+)_T=(\d+)_M=(\d+)_N=(\d+)"
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'model_dim': int(match.group(1)),
            'iterations': int(match.group(2)),
            'memory_length': int(match.group(3)),
            'hanoi_disks': int(match.group(4))
        }
    return None


def encode_hanoi_state(state, num_pegs=3):
    """Encode Tower of Hanoi state for model input."""
    # state: [disk1_peg, disk2_peg, ..., diskN_peg]
    # Returns one-hot encoded representation
    num_disks = len(state)
    encoded = torch.zeros(num_disks, num_pegs)
    
    for i, peg in enumerate(state):
        encoded[i, peg] = 1.0
    
    return encoded.flatten()


def decode_hanoi_state(encoded_state, num_disks, num_pegs=3):
    """Decode model output back to Tower of Hanoi state."""
    # Reshape to [num_disks, num_pegs]
    reshaped = encoded_state.view(num_disks, num_pegs)
    
    # Get peg assignment for each disk
    state = torch.argmax(reshaped, dim=1)
    
    return state


def is_valid_hanoi_state(state):
    """Check if a Tower of Hanoi state is valid (larger disks below smaller ones)."""
    num_disks = len(state)
    pegs = {}
    
    # Group disks by peg
    for disk, peg in enumerate(state):
        peg = peg.item() if hasattr(peg, 'item') else peg
        if peg not in pegs:
            pegs[peg] = []
        pegs[peg].append(disk)
    
    # Check each peg for valid ordering
    for peg, disks in pegs.items():
        if len(disks) <= 1:
            continue
        
        # Sort disks on this peg (smaller disk number = smaller disk)
        disks.sort()
        
        # Check if they are in ascending order (smallest on top)
        for i in range(len(disks) - 1):
            if disks[i] > disks[i + 1]:
                return False
    
    return True


def get_optimal_moves(num_disks):
    """Get the optimal number of moves for Tower of Hanoi with num_disks."""
    return (2 ** num_disks) - 1