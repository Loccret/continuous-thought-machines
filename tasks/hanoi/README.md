# Tower of Hanoi

This task implements the classic Tower of Hanoi puzzle where the goal is to move N disks from one peg to another, following the rules:
1. Only one disk can be moved at a time
2. Only the top disk from a peg can be moved
3. A disk cannot be placed on top of a smaller disk

## Task Description

The Tower of Hanoi dataset generates puzzles with N disks and challenges the model to learn the optimal sequence of moves. The state representation consists of the peg position for each disk (0, 1, or 2 representing the three pegs).

### State Representation
- Input: `[disk1_peg, disk2_peg, ..., diskN_peg]` where disk1 is the smallest
- Target: Next optimal state in the solution sequence
- Goal: Move all disks from peg 0 to peg 2

### Key Features
- Configurable number of disks (N)
- Optimal solution generation using recursive algorithm
- State validation to ensure legal moves
- Visualization of disk positions and model predictions

## Training

To run Tower of Hanoi training, use the bash scripts from the root level of the repository. For example, to train a 75-iteration, 25-memory-length CTM on 3-disk puzzles:

```bash
bash tasks/hanoi/scripts/train_ctm_75_25_3disks.sh
```

### Available Scripts
- `train_ctm_75_25_3disks.sh` - Train CTM with 3 disks
- `train_ctm_75_25_4disks.sh` - Train CTM with 4 disks  
- `train_ctm_75_25_5disks.sh` - Train CTM with 5 disks
- `train_lstm_3disks.sh` - Train LSTM baseline with 3 disks

### Training Arguments

Key arguments for training:
- `--hanoi_disks`: Number of disks in the puzzle (default: 3)
- `--iterations`: Number of CTM internal iterations (default: 75)
- `--memory_length`: Length of memory for NLMs (default: 25)
- `--d_model`: Model dimension (default: 1024)
- `--batch_size`: Training batch size (default: 128)
- `--num_epochs`: Number of training epochs (default: 50)

## Analysis

To run analysis on trained models:

```bash
python -m tasks.hanoi.analysis.run --log_dir <PATH_TO_LOG_DIR>
```

### Metrics
- **Solution Validity**: Percentage of generated solutions that follow Hanoi rules
- **Optimality**: Percentage of solutions using minimum number of moves
- **Success Rate**: Percentage of puzzles solved correctly
- **Average Moves**: Average number of moves in generated solutions

## Problem Complexity

The Tower of Hanoi problem scales exponentially:
- 3 disks: 7 optimal moves
- 4 disks: 15 optimal moves
- 5 disks: 31 optimal moves
- N disks: 2^N - 1 optimal moves

This makes it an excellent benchmark for testing sequential reasoning and planning capabilities in neural networks.

## Visualization

The training script generates visualization GIFs showing:
- Current disk positions on pegs
- Model predictions vs target states
- Neural activation patterns
- Attention weights (for attention-based models)
- Prediction confidence over time

Example output files:
- `hanoi_visualization.gif` - Step-by-step solution visualization
- `training_curves.png` - Loss and accuracy plots
- `solution_analysis.json` - Detailed performance metrics