# Simulating Federated Learning using FedAvg

**Mohammed Aftaab Bin Usman**
Department of Computer Science, Illinois Institute of Technology; Chicago, IL, United States

## Abstract

This project implements a Federated Learning (FL) simulation for image classification using non-IID data partitions across multiple simulated devices. A ResNet-18-based neural network is trained concurrently on client datasets using a `ThreadPoolExecutor` to emulate distributed learning. The final global model is deployed via a Flask-based web application, containerized using Docker, to allow inference requests over the internet. Results demonstrate the effectiveness of FL for collaborative learning while preserving data privacy.

**Index Terms:** Federated Learning, Non-IID Data, ResNet-18, Docker, Flask, Model Deployment, CIFAR-10

## 1. Introduction

Deep learning models have shown remarkable promise in the image classification domain, but training such models typically requires aggregating large amounts of data at a centralized location. This raises privacy concerns in some applications, since sensitive data from a user's device might need to be shared.

Federated learning (FL) resolves this by allowing multiple devices to train a global model without ever needing to transfer their raw data to a central server [1]. Each device trains locally on its own data, sending only model updates to the server, which then combines them.

To make the simulation more realistic, non-IID splits of data across clients were intentionally created to simulate class imbalance across devices. After training the global model, it's deployed as a simple web application that performs inference on new images, demonstrating how FL can be applied in a practical context.

## 2. Methodology

### 2.1 FL Simulation

A simple FL setup was built from scratch using PyTorch, simulating multiple nodes each with a unique local dataset and training process. The simulation follows the standard federated averaging (FedAvg) approach — multiple clients train local models on their own data, and a central server aggregates these models to form a global model. This approach is widely used in FL research for its simplicity and effectiveness [2].

The CIFAR-10 dataset was partitioned across 64 clients in a non-IID manner, with ResNet-18 as the backbone for each client's local training. Client model parameters are sent to a central server and averaged to update the global model. Each client's accuracy and per-batch training loss were recorded during training, and the global model was evaluated on a held-out validation set after every communication round.

### 2.2 Web Application

A simple web application was built using Flask, allowing users to interact with the trained model and perform inference. Docker containerizes the web application to ensure reproducibility and enable deployment on Chameleon Cloud. The modular design separates the core FL training script from the web interface, making updates and redeployment easier.

## 3. Experimental Design & Results

The FL simulation was run on 64 clients with non-IID partitions of CIFAR-10. Each client trained its own instance of ResNet-18 on its own data for two epochs per communication round, over a total of 50 communication rounds. FedAvg was used to aggregate client updates. Each client's batch-level training loss and accuracy were recorded, and the global model was tested on a separate validation set after each round.

### Non-IID data distribution

<img src="results/figures/non_iid_split.png" width="500">

*Fig. 1 — Non-IID distribution of data across 64 clients.*

Client data was partitioned using a Dirichlet distribution with concentration parameter α = 0.5, adding a moderate degree of heterogeneity among clients. A lower α (e.g., 0.1) creates a more skewed, imbalanced distribution, increasing the difficulty of global convergence since clients specialize heavily in certain classes. A higher α (e.g., 1.0+) makes the data more uniform, leading to smoother convergence and potentially higher validation accuracy. Increasing local epochs or total communication rounds can also improve convergence, but risks overfitting if local client updates dominate the global aggregation.

The main metrics tracked were:
- **Train Loss and Accuracy** — average loss and accuracy across all clients, per round
- **Validation Loss and Accuracy** — global model performance on a held-out validation set after each round

Training loss decreased steadily across communication rounds while validation accuracy increased, indicating effective learning across clients despite the non-IID data. After the final communication round, the global model achieved a **validation loss of 0.7111** and a **validation accuracy of 76.67%**, demonstrating that FedAvg can handle heterogeneous client data successfully.

### Accuracy and loss curves

<img src="results/figures/accuracy_curve.png" width="400"> <img src="results/figures/loss_curve.png" width="400">

*Fig. 2/3 — Training vs. validation accuracy and loss across communication rounds.*

Train accuracy rapidly increases and stabilizes near 95%, clearly outperforming validation accuracy, which plateaus around 77.5% — indicating the model overfits to the training data. Both loss curves decrease as training progresses: train loss drops rapidly to nearly 0.15, reflecting successful optimization, while validation loss stabilizes earlier at a higher value near 0.7.

### Inference demo

A test image (dog) was submitted to the `/predict` endpoint of the web service (`http://127.0.0.1:8000`) and correctly classified as class label 5 — which corresponds to "dog" in CIFAR-10.

These results show that the FL training was simulated successfully, approximating the performance of centralized training while preserving data locality and client privacy — confirming that collaborative learning via FedAvg is effective even with heterogeneous client data.

## 4. Conclusions

The goal of this project was to explore FL for image classification on non-IID data and demonstrate how this approach can be used to deploy models for broader access. The motivation ties to the growing need to train models collaboratively across distributed data sources without compromising data privacy. This simulation, based on the federated averaging approach, shows that a good global model can still be trained even when data isn't uniformly distributed among clients.

Potential directions for future work:
- Experimenting with more sophisticated aggregation algorithms like **FedProx** or **FedNova** to better handle client heterogeneity
- Exploring privacy-preserving techniques such as **differential privacy** or **secure aggregation** to improve data confidentiality
- Applying this framework to more complex datasets and larger-scale experiments on cloud platforms, to better understand the robustness and scalability of federated learning

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *AISTATS*, 2017.

[2] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated learning: Challenges, methods, and future directions," *IEEE Signal Processing Magazine*, vol. 37, no. 3, pp. 50–60, 2020.
