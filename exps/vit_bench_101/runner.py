import logging
import os  # Added for log directory creation
import sys

import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from exps.mq_bench_101.dataset import ZcDataset
from exps.mq_bench_101.parzc import BaysianMLPMixer
from piconas.core.losses.diffkd import diffkendall
from piconas.core.losses.pair_loss import pair_loss
from piconas.utils.rank_consistency import kendalltau, pearson, spearman

# Create a log directory if it doesn't exist
log_dir = './logdir'
os.makedirs(log_dir, exist_ok=True)

# Set up logging to save logs to './logdir'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(
    filename=os.path.join(
        log_dir,
        'training_vit_bench_101_231107_13.log'),  # Save logs to a file
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p')

# Create a console handler to print logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

json_path = './checkpoints/vit-bench-layerwise-zc.json'

# Create an instance of the custom dataset
dataset = ZcDataset(json_path)

# Split the data into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

# Create data loaders for batch processing
batch_size = 35
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Create an instance of the MLP model and define the loss function and optimizer
mlp_model = BaysianMLPMixer(
    input_dim=58,
    sequence_length=765 - 765 % 16,
    patch_size=16,
    dim=1600,
    depth=5,
    emb_out_dim=1,
    expansion_factor=4,
    expansion_factor_token=0.5,
    dropout=0.18)

loss_function = diffkendall
optimizer = optim.Adam(mlp_model.parameters(), lr=0.002)

# Move the model to GPU if available
if torch.cuda.is_available():
    mlp_model.cuda()

# Train the model
num_epochs = 296
logging.info('Start training...')
for epoch in range(num_epochs):
    mlp_model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        optimizer.zero_grad()
        y_pred = mlp_model(batch_x)
        loss = diffkendall(y_pred.squeeze(-1), batch_y)
        loss.backward()
        optimizer.step()

        # Print the loss every 2 iteration
        if batch_idx % 5 == 0:
            logging.info(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}'
            )

    if (epoch + 1) % 1 == 0:
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
        y_pred.extend(batch_pred.cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())

# Calculate the mean squared error (MSE) loss between the predicted and actual values
mse_loss = mean_squared_error(y_true, y_pred)
logging.info(f'MSE Loss: {mse_loss:.4f}')

# Calculate the kendall tau, spearman, pearson correlation coefficients
logging.info(f'Kendall tau: {kendalltau(y_true, y_pred)}')
logging.info(f'Spearman: {spearman(y_true, y_pred)}')
logging.info(f'Pearson: {pearson(y_true, y_pred)[0]}')
