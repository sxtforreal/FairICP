import torch

# Set device cuda for GPU if it is available, otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
min_epochs = 3
max_epochs = 50
lr = 1e-4
emb_dim = 768
hidden_dim1 = 300
hidden_dim2 = 50
num_classes = 2
dropout_ratio = 0.2

# Dataset
batch_size = 20

# Logger
log_every_n_steps = 10
ckpt_every_n_steps = 10

# Compute related
accelerator = "gpu"
devices = 1
precision = "16-mixed"
