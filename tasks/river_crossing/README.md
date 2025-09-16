# River Crossing

This task implements the classic River Crossing puzzle where N pairs of entities need to cross a bridge, but the bridge can only hold a limited number at a time and they need a torch/flashlight to cross safely.

## Task Description

The River Crossing dataset generates puzzles with N pairs and challenges the model to learn the optimal sequence of moves. Each pair consists of two entities that must cross together, and only a limited number can cross at a time.

### Classic Rules
1. The bridge can only hold 2 people at a time
2. They need a torch to cross, and only one torch is available
3. When crossing, they must go at the speed of the slowest person
4. Someone must bring the torch back to the other side (except for the final crossing)

### State Representation
- Input: `[pair1_pos, pair2_pos, ..., pairN_pos, bridge_pos]`
- Values: 0 = left side, 1 = right side
- Goal: Move all pairs from left side (0) to right side (1)

### Key Features
- Configurable number of pairs (N)
- Optimal solution generation using strategic algorithms
- State validation to ensure legal moves
- Visualization of crossing progress and model predictions
- Support for different crossing strategies

## Training

To run River Crossing training, use the bash scripts from the root level of the repository. For example, to train a 75-iteration, 25-memory-length CTM on 3-pair puzzles:

```bash
bash tasks/river_crossing/scripts/train_ctm_75_25_3pairs.sh
```

### Available Scripts
- `train_ctm_75_25_3pairs.sh` - Train CTM with 3 pairs
- `train_ctm_75_25_4pairs.sh` - Train CTM with 4 pairs  
- `train_ctm_75_25_5pairs.sh` - Train CTM with 5 pairs
- `train_lstm_3pairs.sh` - Train LSTM baseline with 3 pairs

### Training Arguments

Key arguments for training:
- `--river_pairs`: Number of pairs in the puzzle (default: 3)
- `--iterations`: Number of CTM internal iterations (default: 75)
- `--memory_length`: Length of memory for NLMs (default: 25)
- `--d_model`: Model dimension (default: 1024)
- `--batch_size`: Training batch size (default: 128)
- `--num_epochs`: Number of training epochs (default: 50)

## Analysis

To run analysis on trained models:

```bash
python -m tasks.river_crossing.analysis.run --log_dir <PATH_TO_LOG_DIR>
```

### Metrics
- **Solution Validity**: Percentage of generated solutions that follow crossing rules
- **Optimality**: Percentage of solutions using minimum number of moves
- **Success Rate**: Percentage of puzzles solved correctly
- **Average Moves**: Average number of moves in generated solutions
- **Bridge Utilization**: How efficiently the bridge capacity is used

## Problem Complexity

The River Crossing problem scales with the number of pairs:
- 1 pair: 1 move (trivial case)
- 2 pairs: 5 moves
- 3 pairs: 9 moves  
- 4 pairs: 13 moves
- N pairs: 4N - 3 moves (optimal strategy)

### Strategic Considerations
The puzzle requires strategic thinking about:
- **Order of crossing**: Which pairs should go first
- **Return strategy**: Who should bring the torch back
- **Timing optimization**: Minimizing total crossing time
- **Capacity management**: Efficiently using bridge capacity

## Crossing Strategies

### Two-Fastest Strategy
1. Two fastest pairs cross together
2. Fastest returns with torch
3. Two slowest cross together
4. Second fastest returns with torch
5. Repeat until all have crossed

### Sequential Strategy
1. Fastest pair crosses with each other pair
2. Fastest returns each time
3. Simple but often suboptimal

## Visualization

The training script generates visualization GIFs showing:
- Current positions of all pairs on either side
- Bridge position and torch location
- Model predictions vs target states
- Neural activation patterns
- Attention weights (for attention-based models)
- Prediction confidence over time

Example output files:
- `river_crossing_visualization.gif` - Step-by-step solution visualization
- `training_curves.png` - Loss and accuracy plots
- `strategy_analysis.png` - Visualization of crossing strategies
- `solution_analysis.json` - Detailed performance metrics

## Problem Variants

The implementation can be extended to support various River Crossing variants:
- **Different crossing times**: Each pair has different crossing speeds
- **Variable bridge capacity**: Bridge can hold different numbers of people
- **Multiple torches**: More than one torch available
- **Asymmetric pairs**: Pairs with different characteristics
- **Time constraints**: Maximum total crossing time allowed