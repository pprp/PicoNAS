# use argparse to get search space
import argparse
import logging
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from exps.mq_bench_101.parzc import BaysianMLPMixer
from exps.nds.zc_dataset import ZcDataset
from piconas.core.losses.diffkd import diffkendall
from piconas.core.losses.pair_loss import pair_loss
from piconas.utils.rank_consistency import kendalltau, pearson, spearman, spearman_top_k

parser = argparse.ArgumentParser()
parser.add_argument('--search_space', type=str, default='Amoeba')
args = parser.parse_args()

SEARCH_SPACE = {'Amoeba', 'DARTS', 'ENAS', 'NASNet', 'PNAS'}

assert (
    args.search_space in SEARCH_SPACE
), f'args.search_space: {args.search_space} not in {SEARCH_SPACE}'

# Create a log directory if it doesn't exist
log_dir = './logdir/nds'
os.makedirs(log_dir, exist_ok=True)

# Set up logging to save logs to './logdir'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(
    filename=os.path.join(
        log_dir, f'training_hpo_NDS_{args.search_space}_NDSKD.log'
    ),  # Save logs to a file
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p',
)

# Create a console handler to print logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

json_path = 'checkpoints/nds_data/nds_ss_layerwise_zc.json'


# Define a function to perform hyperparameter optimization
def objective(trial):
    # Sample hyperparameters to optimize
    patch_size = 16
    max_sequence_length = 4096  # Maximum sequence length
    sequence_length = trial.suggest_int(
        'sequence_length', patch_size, max_sequence_length
    )
    # Ensure sequence_length is divisible by patch_size
    sequence_length -= sequence_length % patch_size
    dim = trial.suggest_int('dim', 256, 4096)
    depth = trial.suggest_int('depth', 2, 8)  # Use a step value of 1
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    num_epochs = trial.suggest_int('num_epochs', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Create an instance of the custom dataset
    dataset = ZcDataset(args.search_space, json_path)

    # Split the data into training and testing sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders for batch processing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    if args.search_space == 'Amoeba':
        input_dim = 2832
    elif args.search_space == 'DARTS':
        input_dim = 2000
    elif args.search_space == 'ENAS':
        input_dim = 2752
    elif args.search_space == 'NASNet':
        input_dim = 2520
    elif args.search_space == 'PNAS':
        input_dim = 2912

    # Create an instance of the MLP model and define the loss function and optimizer
    mlp_model = BaysianMLPMixer(
        input_dim=input_dim,
        sequence_length=sequence_length,
        patch_size=16,
        dim=dim,
        depth=depth,
        emb_out_dim=1,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=dropout,
    )

    loss_function = pair_loss
    optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    if torch.cuda.is_available():
        mlp_model.cuda()

    # Train the model
    logging.info('Start training...')
    for epoch in range(num_epochs):
        mlp_model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            optimizer.zero_grad()
            y_pred = mlp_model(batch_x)
            loss = loss_function(y_pred.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            logging.info(
                f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    logging.info('Training completed!')

    # Evaluate the model
    mlp_model.eval()
    y_pred = []
    y_true = []
    logging.info('Start evaluation...')
    for batch_x, batch_y in test_loader:
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        with torch.no_grad():
            batch_pred = mlp_model(batch_x)
            y_pred.extend(batch_pred.cpu().numpy().tolist())
            y_true.extend(batch_y.cpu().numpy().tolist())

    # Convert to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).squeeze(-1)

    # Calculate the kendall tau, spearman, pearson correlation coefficients
    logging.info(f'Kendall tau: {kendalltau(y_true, y_pred)}')
    logging.info(f'Spearman: {spearman(y_true, y_pred)}')
    logging.info(f'Pearson: {pearson(y_true, y_pred)}')

    # Calculate Spearman topk
    sp_list = spearman_top_k(y_true, y_pred)
    logging.info(f'Spearman topk@20%: {sp_list[0]}')
    logging.info(f'Spearman topk@50%: {sp_list[1]}')
    logging.info(f'Spearman topk@100%: {sp_list[2]}')

    return np.mean(sp_list)


# Create an Optuna study
study = optuna.create_study(direction='maximize')

# Optimize hyperparameters
study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

# Print the best hyperparameters and corresponding loss
logging.info('Best trial:')
best_trial = study.best_trial
logging.info(f'  Value: {best_trial.value:.4f}')
logging.info('  Params:')
for key, value in best_trial.params.items():
    logging.info(f'    {key}: {value}')
