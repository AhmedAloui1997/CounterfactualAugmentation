import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, T, Y, epsilon=0.01):
        self.X = X
        self.T = T
        self.Y = Y
        self.epsilon = epsilon

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]
        t1 = self.T[idx]
        y1 = self.Y[idx]

        similar_indices = np.where((self.T == t1) & (np.abs(self.Y - y1) < self.epsilon))[0]
        dissimilar_indices = np.where((self.T == t1) & (np.abs(self.Y - y1) > self.epsilon))[0]

        similar_idx = np.random.choice(similar_indices)
        dissimilar_idx = np.random.choice(dissimilar_indices)

        x2_similar = self.X[similar_idx]
        x2_dissimilar = self.X[dissimilar_idx]

        return x1, x2_similar, x2_dissimilar

# Define the model for our task
class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ContrastiveLearningModel, self).__init__()
        # Define the architecture of the network
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        # Compute the embeddings of the input
        return self.embedding(x)

# Define the contrastive loss function
def contrastive_loss(embeddings1, embeddings2, labels, margin):
    # Compute the Euclidean distances between the embeddings
    distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
    # Compute the loss based on the distances and the labels
    losses = (1 - labels) * torch.pow(distances, 2) + labels * torch.pow(torch.relu(margin - distances), 2)
    # Return the average loss
    return losses.mean()

# Define the function to train the model
def train_model(X, T, Y, epsilon, input_dim, embedding_dim, batch_size, num_epochs, learning_rate, margin):
    # Create the custom dataset and the dataloader
 

    dataset = CustomDataset(X, T, Y, epsilon)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize the model and the optimizer
    model = ContrastiveLearningModel(input_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Get the data points for this batch
            x1, x2_similar, x2_dissimilar = batch
            # Convert them to float for computation
            x1, x2_similar, x2_dissimilar = x1.float(), x2_similar.float(), x2_dissimilar.float()
            #x1, x2_similar, x2_dissimilar = [x.to(device) for x in batch]
            #x1, x2_similar, x2_dissimilar = x1.to(device), x2_similar.to(device), x2_dissimilar.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Compute the embeddings for the data points
            embeddings1 = model(x1)
            embeddings2_similar = model(x2_similar)
            embeddings2_dissimilar = model(x2_dissimilar)

            current_batch_size = x1.size(0)
            loss_similar = contrastive_loss(embeddings1, embeddings2_similar, labels=torch.zeros(current_batch_size), margin=margin)
            loss_dissimilar = contrastive_loss(embeddings1, embeddings2_dissimilar, labels=torch.ones(current_batch_size), margin=margin)
            loss = 0.5 * (loss_similar + loss_dissimilar)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model
