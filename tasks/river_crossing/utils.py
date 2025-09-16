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
    # inputs shape: [batch_size, river_pairs + 1]
    # For CTM/LSTM, we need: [batch_size, seq_len, input_dim]
    batch_size = inputs.size(0)
    
    # Add sequence dimension and expand input dimension
    inputs = inputs.unsqueeze(1)  # [batch_size, 1, river_pairs + 1]
    
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
    # Example: "ctm_river_D=1024_T=75_M=25_N=3"
    pattern = r"ctm_river_D=(\d+)_T=(\d+)_M=(\d+)_N=(\d+)"
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'model_dim': int(match.group(1)),
            'iterations': int(match.group(2)),
            'memory_length': int(match.group(3)),
            'river_pairs': int(match.group(4))
        }
    return None


def encode_river_state(state):
    """Encode River Crossing state for model input."""
    # state: [pair1_pos, pair2_pos, ..., pairN_pos, bridge_pos]
    # Already in binary format (0 = left, 1 = right)
    return state.float()


def decode_river_state(encoded_state, num_pairs):
    """Decode model output back to River Crossing state."""
    # Round to nearest binary values
    state = torch.round(encoded_state).long()
    state = torch.clamp(state, 0, 1)  # Ensure binary values
    
    return state


def is_valid_river_state(state, bridge_pos=None):
    """Check if a River Crossing state is valid."""
    # Basic validation: all values should be 0 or 1
    if bridge_pos is None:
        bridge_pos = state[-1].item() if hasattr(state[-1], 'item') else state[-1]
    
    # Check if all values are binary
    for pos in state:
        val = pos.item() if hasattr(pos, 'item') else pos
        if val not in [0, 1]:
            return False
    
    return True


def get_optimal_moves_river(num_pairs):
    """Get the optimal number of moves for River Crossing with num_pairs."""
    if num_pairs == 1:
        return 1
    else:
        return 4 * num_pairs - 3


def simulate_river_crossing_move(state, pairs_to_move):
    """Simulate a move in the River Crossing puzzle."""
    new_state = state.clone()
    
    # Get current bridge position
    bridge_pos = new_state[-1].item()
    
    # Move specified pairs
    for pair_idx in pairs_to_move:
        if pair_idx < len(new_state) - 1:
            # Toggle pair position
            new_state[pair_idx] = 1 - new_state[pair_idx]
    
    # Bridge moves to opposite side
    new_state[-1] = 1 - new_state[-1]
    
    return new_state


def check_river_crossing_solution(initial_state, moves):
    """Check if a sequence of moves solves the River Crossing puzzle."""
    current_state = initial_state.clone()
    target_state = torch.ones_like(initial_state)  # All on right side
    
    for move in moves:
        current_state = simulate_river_crossing_move(current_state, move)
        
        # Check if state is valid after move
        if not is_valid_river_state(current_state):
            return False
    
    # Check if final state matches target
    return torch.equal(current_state, target_state)


def get_crossing_time(pairs_times, pairs_to_move):
    """Get the time taken for a group to cross (max time among the group)."""
    if not pairs_to_move:
        return 0
    
    times = [pairs_times[i] for i in pairs_to_move if i < len(pairs_times)]
    return max(times) if times else 0


def optimize_river_crossing_strategy(num_pairs, crossing_times=None):
    """Generate an optimized strategy for River Crossing puzzle."""
    if crossing_times is None:
        # Default: each pair takes increasing time to cross
        crossing_times = list(range(1, num_pairs + 1))
    
    if num_pairs == 1:
        return [(0,)]  # Single pair crosses
    
    # Two fastest pairs strategy
    moves = []
    
    # Two fastest cross
    moves.append((0, 1))
    
    # Fastest returns
    moves.append((0,))
    
    # Two slowest cross
    for i in range(2, num_pairs):
        moves.append((i, num_pairs - 1))
        if i < num_pairs - 1:  # Not the last pair
            moves.append((1,))  # Second fastest returns
    
    return moves