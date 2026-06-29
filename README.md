A character-level language model for name generation built following along Andrej Karpathy's Neural Networks: Zero to Hero series. Implementation is done in two ways, counting and single-layer neural net. 

## What it does
Trains on a dataset of names (names.txt) and generates new ones from the learned bigram neural net. 

## Models
Counting Model - creates a 27x27 bigram count matrix, each cell is then normalized by row sum to create a probability distribution
Single-layer Neural Net Model - a single layer trained using gradient descent with the loss function defined as NLL. This converges to the counting model. 

## Run
```bash
python3.14 -m venv .venv 
source .venv/bin/activate
pip install torch
# CPU version is fine
jupyter notebook bigram.ipynb
```

## Notes
Additional notes in [notes/bigram.md](notes/bigram.md) 