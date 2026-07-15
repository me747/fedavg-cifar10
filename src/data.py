import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import seaborn as sns
import collections

NUM_CLIENTS = 64 # no. of simulated devices
BATCH_SIZE = 32 # batch size for each client during training


# *ToTensor() for converting the CIFAR-10 images (numpy arrays with range [0, 255]) to Pytorch Tensor with values scaled to [0, 1]
# *Normalize() each of the channels (R, G, B) by subtracting the mean = 0.5 and dividing by std = 0.5
# might compute the actual mean and std for CIFAR-10 later, but for now 0.5 is decent
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# loading CIFAR-10 dataset
train_data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
test_data = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

# checking no. of training and testing samples available 
print(f"Training Samples: {len(train_data)}, Test Samples: {len(test_data)}")

def split_data_non_iid(dataset, NUM_CLIENTS, num_labels = 10, alpha = 0.5):
    
    labels = np.array(dataset.targets) # *dataset.targets is a list of class labels for each image in CIFAR-10, convert into a NumPy array for filtering
    
    # a dictionary where each key is client id, and its value(list) holds the dataset labels assigned to it
    client_indices = {}
    for i in range(NUM_CLIENTS):
        client_indices[i] = []
    
    # loop through each class
    for i in range(num_labels):
            label_indices = np.where(labels == i)[0] # *np.where returns a tuple of arrays, so using [0] returns the 1st array.
            # 1st array contains indices for label 0
            np.random.shuffle(label_indices) # randomizing selection 

            # using dirichlet distribution to ensure non-iid distribution of data i.e. it determines the proportion of class labels that go to each client
            proportions = np.random.dirichlet([alpha] * NUM_CLIENTS)

            # converting proportions to integer value no. of samples per client
            samples_per_client = (proportions * len(label_indices)).astype(int)

            k = 0 # pointer to keep track of current position in label_indices
            for cid, n_samples in enumerate(samples_per_client):
                client_indices[cid].extend(label_indices[k:k + n_samples]) # extend current slice of indices to current client in the dict
                k += n_samples # remaining samples to assign

            # incase theres any leftover samples (distribute leftover samples over the clients cyclically)
            leftover = len(label_indices) - k
            for i in range(leftover):
                client_indices[i % NUM_CLIENTS].append(label_indices[k + i])

    # finally create a new dataset object (PyTorch Subsets) that contains only the selected samples for each client
    client_data = {}
    for i, indices in client_indices.items():
         client_data[i] = Subset(dataset, indices)

    return client_data      

client_datasets = split_data_non_iid(train_data, NUM_CLIENTS)

client_loaders = {}
for i, ds in client_datasets.items():
     client_loaders[i] = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

if __name__ == "__main__":
   
#  create a dataframe with client IDs as rows and class labels as columns to visualize
   data = []
   for cid, ds in client_datasets.items():
       client_labels = [train_data.targets[idx] for idx in ds.indices]
       label_counts = collections.Counter(client_labels)
     
       row = []
       for i in range(10):
          row.append(label_counts.get(i, 0))
       data.append(row)

   df = pd.DataFrame(data, columns=[f"Class {i}" for i in range(10)])
   df.index.name = "Client ID"

   plt.figure(figsize=(12, 6))
   sns.heatmap(df, cmap="viridis", cbar_kws={'label': 'Number of Samples'})
   plt.title(f"Non-IID Data Distribution Across {NUM_CLIENTS} Clients")
   plt.show()