import pandas as pd
import os
import copy
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import client_loaders, test_data, BATCH_SIZE, NUM_CLIENTS
from model import ResNet18_Fed

# to average model weights from all clients (FedAvg)
def avg_weights(weights_list):
    # copy of the first client's weights to avoid modifying original
    avg = copy.deepcopy(weights_list[0])
    
    # loop over each parameter in the model, sum over the same param. from all clients and then divide by no. of clients to avg
    for k in avg.keys():
        for i in range(1, len(weights_list)):
            avg[k] += weights_list[i][k]
        avg[k] = avg[k] / len(weights_list)
    return avg

# local training 
def local_train(client_id, global_state, loader, local_epochs, round_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initializing local model and loading the current global model weights
    model = ResNet18_Fed().to(device)
    model.load_state_dict(global_state)
    
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # optimizer

    # store logs for each batch
    logs = []

    # training loop for specified no. of epochs 
    for epoch in range(local_epochs):
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)  # move data to device
            optimizer.zero_grad()  # clear prior gradients
            outputs = model(x)  # forward pass
            loss = criterion(outputs, y)  # calculate loss
            loss.backward()  # backpropagate
            optimizer.step()  # update model weights

            # calculate batch accuracy
            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            # save the batch metrics into the logs list
            logs.append({
                "time": time.time(),       # timestamp of batch
                "round": round_num,        # communication round
                "batch_num": batch_idx,    # batch index
                "client_id": client_id,    # which client
                "train_loss": loss.item(), # loss value
                "train_acc": acc           # accuracy
            })

    # return the updated local model weights and training logs
    return copy.deepcopy(model.state_dict()), logs

# evaluate global model on test/validation data
def evaluate(model, test_dataset):
    device = torch.device("cpu")
    model.to(device).eval() 
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_loss, correct, total = 0.0, 0, 0  
    with torch.no_grad():  # no gradient calculation needed
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)  # sum up batch loss
            _, predicted = torch.max(outputs, 1)  # predicted labels
            correct += (predicted == y).sum().item()  # count no. of correct predictions
            total += y.size(0)  # total no. of samples

    # calculate average loss and accuracy
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")
    return avg_loss, acc  

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently using: {device}")

    # initialize global model
    global_model = ResNet18_Fed().to(device)
    global_state = global_model.state_dict()  # get initial model weights

    # FL hyperparameters
    ROUNDS = 50          # no. of communication rounds
    LOCAL_EPOCHS = 2    # no. of local training epochs per client per round

    all_logs = []       # store batch-wise training logs
    round_metrics = []  # store validation metrics per communication round

    # loop over communication rounds
    for r in range(ROUNDS):
        print(f"\n --Communication Round {r+1}--")
        client_states = []  # collect updated client weights

        # train all clients concurrently using threads
        with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = []
            for cid, loader in client_loaders.items():
                # submitting a local training job to each client
                futures.append(
                    executor.submit(local_train, cid, global_state, loader, LOCAL_EPOCHS, r+1)
                )
            # collect results as they finish
            for f in futures:
                state, logs = f.result()
                client_states.append(state)  # save client weights
                all_logs.extend(logs)        # add batch logs to main logs list

        # Aggregate using FedAvg
        global_state = avg_weights(client_states)
        global_model.load_state_dict(global_state)  # update global model

        # evaluate global model on validation/test dataset
        val_loss, val_acc = evaluate(global_model, test_data)
        round_metrics.append({
            "round": r+1,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # save the final global model
    torch.save(global_model.state_dict(), "../results/global_model.pt")
    print("Saved global model to global_model.pt")

    # ensure results folder exists
    os.makedirs("../results", exist_ok=True)
    
    # save logs as CSV files
    pd.DataFrame(all_logs).to_csv("../results/training_logs.csv", index=False)  
    pd.DataFrame(round_metrics).to_csv("../results/round_metrics.csv", index=False)  
    print("Training and validation logs saved to results folder")
