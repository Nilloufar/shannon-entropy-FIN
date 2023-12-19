import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import random
import numpy as np
import collections
import math
from tqdm.auto import tqdm
from scipy.stats import kurtosis
import os

FIN_path = os.path.dirname(__file__)

trained_fin_path = {"mean_400": "/fin_modules/mean/mean_0.002299_n_400.pth",
                    "std_400":"/fin_modules/std/std_0.003824_n_400.pth",
                    "mean_768":"/fin_modules/mean/mean_0.001322_n_768.pth",
                    "mean_1024": "/fin_modules/mean/mean_0.003412_n_1024.pth",

                    }

# Define a neural network that learns to compute the mean
class MeanFIN(nn.Module):
    def __init__(self,dim=400):
        super(MeanFIN, self).__init__()
        self.active_fnc = nn.ReLU()

        self.net_layers= nn.Sequential(nn.Linear(dim, dim * 4),
                                       self.active_fnc,
                                       nn.Linear(dim * 4, dim * 2),
                                       self.active_fnc,
                                       nn.Linear(dim * 2, dim),
                                       self.active_fnc,
                                       nn.Linear(dim, int(dim / 2)),
                                       self.active_fnc,
                                       nn.Linear(int(dim / 2), 1))
    def _init_weights(self):
        for m in self.net_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def forward(self, x):
        x = self.net_layers(x)
        return  x



class STDFIN(nn.Module):
    def __init__(self,dim,mean_model):
        super(STDFIN, self).__init__()

        # Mean Network
        self.mean_net = mean_model
        # Freeze the weights of the mean_net
        for param in self.mean_net.parameters():
            param.requires_grad = False

        # Std Network

        self.dropout = nn.Dropout(p=0.01)
        self.fc1 = nn.Linear(dim+1, dim*4)
        self.fc2 = nn.Linear(dim*4, dim*2)
        self.fc3 = nn.Linear(dim*2, dim)
        self.fc4 = nn.Linear(dim, int(dim/2))
        self.fc5 = nn.Linear(int(dim/2), 1)
        self.active_fnc=torch.nn.functional.relu

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)
        nn.init.normal_(self.fc4.bias, std=1e-6)
        nn.init.normal_(self.fc5.bias, std=1e-6)

    def forward(self, x):
        mean = self.mean_net(x)
        x = torch.cat((x,mean),dim=1)
        x = self.active_fnc(self.fc1(x))
        x = self.active_fnc(self.fc2(x))
        x = self.active_fnc(self.fc3(x))
        x = self.active_fnc(self.fc4(x))
        std = self.fc5(x)


        return std

class MaxFIN(nn.Module):
    def __init__(self,dim):
        super(MaxFIN, self).__init__()
        self.fc1 = nn.Linear(dim, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MinFIN(nn.Module):
    def __init__(self,dim):
        super(MinFIN, self).__init__()
        self.fc1 = nn.Linear(dim, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EntropyFIN(nn.Module):
    def __init__(self, dim=400):
        super(EntropyFIN, self).__init__()
        self.active_fnc = nn.ReLU()
        self.net_layers = nn.Sequential(nn.Linear(dim, dim * 4),
                                        self.active_fnc,
                                        nn.Linear(dim * 4, dim * 2),
                                        self.active_fnc,
                                        nn.Linear(dim * 2, dim),
                                        self.active_fnc,
                                        nn.Linear(dim, int(dim / 2)),
                                        self.active_fnc,
                                        nn.Linear(int(dim / 2), 1))

    def _init_weights(self):
        for m in self.net_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def forward(self, x):
        x = self.net_layers(x)
        return x


class EntropyFINQ(nn.Module):
    def __init__(self, q=2.0, kappa=1.0):
        super(EntropyFINQ, self).__init__()  # Initialize the parent class
        self.q = nn.Parameter(torch.tensor(q))  # Define q as a learnable parameter
        self.kappa = nn.Parameter(torch.tensor(kappa))  # Define kappa as a learnable parameter

    def forward(self, x):
        eps = 1e-8
        x_shape = x.shape
        row_entropies = []

        # Normalize the input using min-max scaling across all values in the tensor
        min_val = torch.min(x)
        max_val = torch.max(x)
        denom = max_val - min_val + eps
        normalized_x = (x - min_val) / denom
        normalized_x = torch.round(normalized_x * 10)

        for idx in range(x_shape[0]):
            # Get unique values and their indices
            unique_values, unique_indices = torch.unique(normalized_x[idx].unsqueeze(0), sorted=True,
                                                         return_inverse=True)
            # Get the counts of each distinct value in the tensor
            counts = torch.bincount(unique_indices.view(-1), minlength=len(unique_values)).float()

            # Compute the Tsallis Entropy
            counts += eps
            counts /= torch.sum(counts, dim=-1, keepdim=True)
            row_entropy = (1 - torch.sum(torch.pow(counts, self.q), dim=-1, keepdim=True)) / (self.q - 1 + eps)

            # Retrun the row entropies as a tensor
            row_entropies.append(row_entropy)

        return torch.cat(row_entropies, dim=0)

class TsallisEntropyFIN(nn.Module):

    def __init__(self, q=2.0, kappa=1.0):
        super(TsallisEntropyFIN, self).__init__()  # Initialize the parent class
        self.q = nn.Parameter(torch.tensor(q))  # Define q as a learnable parameter
        self.kappa = nn.Parameter(torch.tensor(kappa))  # Define kappa as a learnable parameter

    def forward(self, x):
        eps = 1e-8

        x_shape= x.shape
        
        # Flatten the tensor so that all rows are treated as a single batch
        flattened_x = x.view(-1)

        # Get unique values and their indices
        unique_values, unique_indices = torch.unique(flattened_x, sorted=True, return_inverse=True)

        # Get the counts of each distinct value in the tensor
        counts = torch.bincount(unique_indices, minlength=len(unique_values)).float()

        # Reshape counts to match the original tensor shape
        counts = counts.view(x_shape)

        # Compute the Tsallis Entropy
        counts += eps
        counts /= torch.sum(counts, dim=-1, keepdim=True)
        row_entropy = (1 - torch.sum(torch.pow(counts, self.q), dim=-1, keepdim=True)) / (self.q - 1 + eps)


class KurtosisFIN(nn.Module):
    def __init__(self,dim,mean_model,std_model):
        super(KurtosisFIN, self).__init__()

        self.fc1 = nn.Linear(dim+2 , dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim * 2)
        self.fc3 = nn.Linear(dim * 2, dim)
        self.fc4 = nn.Linear(dim, int(dim / 2))
        self.fc5 = nn.Linear(int(dim / 2), 1)
        self.active_fnc = torch.nn.functional.relu
        self.mean_net = mean_model
        self.std_net = std_model



    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)
        nn.init.normal_(self.fc4.bias, std=1e-6)
        nn.init.normal_(self.fc5.bias, std=1e-6)

    def forward(self, x):
        mean = self.mean_net(x)
        std = self.std_net(x)
        x = torch.cat((x, mean,std), dim=1)

        x = self.active_fnc(self.fc1(x))
        x = self.active_fnc(self.fc2(x))
        x = self.active_fnc(self.fc3(x))
        x = self.active_fnc(self.fc4(x))
        x = self.fc5(x)

        return x


class BaseFIN(nn.Module):
    def __init__(self, dim=400):
        super(BaseFIN, self).__init__()
        self.active_fnc = nn.ReLU()
        self.net_layers = nn.Sequential(nn.Linear(dim, dim * 4),
                                        self.active_fnc,
                                        nn.Linear(dim * 4, dim * 2),
                                        self.active_fnc,
                                        nn.Linear(dim * 2, dim),
                                        self.active_fnc,
                                        nn.Linear(dim, int(dim / 2)),
                                        self.active_fnc,
                                        nn.Linear(int(dim / 2), 1))

    def _init_weights(self):
        for m in self.net_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def forward(self, x):
        x = self.net_layers(x)
        return x

class CountFIN(nn.Module):
    def __init__(self,dim):
        super(CountFIN, self).__init__()
        self.fc1 = nn.Linear(dim , dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim * 2)
        self.fc3 = nn.Linear(dim * 2, dim)
        self.fc4 = nn.Linear(dim, int(dim / 2))
        self.fc5 = nn.Linear(int(dim / 2), 10)
        self.active_fnc = torch.nn.functional.relu

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)
        nn.init.normal_(self.fc4.bias, std=1e-6)
        nn.init.normal_(self.fc5.bias, std=1e-6)

    def forward(self, x):
        x = self.active_fnc(self.fc1(x))
        x = self.active_fnc(self.fc2(x))
        x = self.active_fnc(self.fc3(x))
        x = self.active_fnc(self.fc4(x))
        x = self.fc5(x)

        return x



def train_model(model, train_dataloader, val_dataloader, hyperparameters):
    epochs = hyperparameters["epochs"]
    loss_function = hyperparameters["loss_function"]
    base_lr=hyperparameters["lr"]
    optimizer = optim.Adam(FIN_model.parameters(), lr=base_lr, weight_decay=0.0001)
    patience = hyperparameters["patience"]
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience,
                                          verbose=True)  # Reduce learning rate when a metric has stopped improving.
    best_val_loss = float("inf")
    counter = 0
    iter_num=0
    losses = []
    max_iterations = epochs * len(train_dataloader)

    pbar = tqdm(total=(epochs))
    for epoch in range(epochs):
        model = model.cuda()
        model.train()  # Denotes that we are in training mode
        epoch_loss = 0
        for batch, data in enumerate(train_dataloader):
            X, y = data
            model.zero_grad()  # Clear out the gradients from the last `epoch`:
            yhat = model(X.cuda())  # Compute the output of the model, yhat, given x
            loss = loss_function(yhat.cuda(), y.cuda())  # Compute the loss
            loss.backward()  # Compute the gradients of the loss with respect to the model parameters
            optimizer.step()  # Adjust the paramter values in the direction of the gradients
            epoch_loss += loss.item()

        losses.append(epoch_loss)  # Save the loss at each interation for plotting

        # Evaluate the model on the validation set
        val_loss, _ = eval_model(model, val_dataloader, loss_function)

        # Update learning rate with ReduceLROnPlateau scheduler
        # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        # # lr_ = base_lr
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_
        # iter_num+=1
        # scheduler_plateau.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >=  patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # print(
        #     f"Epoch [{epoch + 1}/{epochs}], Loss: {losses[-1]:.8f}, Val Loss: {val_loss:.8f}, LR: {scheduler_plateau.optimizer.param_groups[0]['lr']:.6f}")

        pbar.set_postfix(epoch =epoch,loss=losses[-1] ,val_loss=val_loss.item(), lr=optimizer.param_groups[0]["lr"])
        pbar.update()
    return model, losses


def eval_model(model, val_loader, loss_function):
    model.eval()  # Denotes that we are in evaluation mode
    val_loss = 0
    y_hat = []
    with torch.no_grad():  # Temporarily turn off gradient calculation
        for batch, data in enumerate(val_loader):
            X_val, y_val = data
            # Calculate validation loss
            val_outputs = model(X_val.cuda())
            val_loss += loss_function(val_outputs.cuda(), y_val.cuda())
            y_hat += val_outputs.squeeze().cpu().numpy().tolist()

    return val_loss, y_hat


class FINDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return (self.y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].float()


def create_random_data(num_samples, dim, fin_type="mean"):
    if fin_type == "mean":
        X = torch.rand(num_samples, dim)
        y = X.mean(dim=1, keepdim=True)

    elif fin_type == "shanon_entropy":
        X = torch.rand(num_samples, dim)
        nbins = 10
        n_samples, n_features = X.shape
        y = torch.zeros(n_samples)

        for i in range(n_samples):
            # Discretize x into bins
            x_discrete = (X[i, :] * (nbins - 1)).round().int()

            # Estimate the probability distribution of x
            counts = torch.bincount(x_discrete, minlength=nbins)
            probs = counts.float() / counts.sum()

            # Calculate the Shannon entropy
            log_probs = torch.log2(probs + 1e-10)  # Add a small constant to avoid log(0)
            y[i] = -torch.sum(probs * log_probs)
        y=y.unsqueeze(1)
        # X = [np.random.rand(dim) for i in range(num_samples)]
        # y = [shanon_entropy(X[i]) for i in range(len(X))]
        #
        # X = torch.tensor(np.array(X))
        # y = torch.tensor(np.array(y)).unsqueeze(1)

    elif fin_type == "max":
        X = torch.rand(num_samples, dim)
        y = X.max(dim=1, keepdim=True)[0]

    elif fin_type == "std" or fin_type == "mean_std":
        X = torch.rand(num_samples, dim)
        y = X.std(dim=1, keepdim=True)

    elif fin_type == "min":
        X = torch.rand(num_samples, dim)
        y = X.max(dim=1, keepdim=True)[0]

    elif fin_type == "kurtosis":
        X = torch.rand(num_samples, dim)
        # Convert the input tensor to a NumPy array
        input_array = X.numpy()
        # Compute the Kurtosis for each row using scipy's kurtosis function
        y = np.apply_along_axis(kurtosis, axis=1, arr=input_array)
        # Convert the NumPy array back to a PyTorch tensor
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    elif fin_type == "tsallis_entropy":
        q = 1.1
        eps = 1e-8
        y = []
        X = torch.rand(num_samples, dim)

        # Normalize the input using min-max scaling across all values in the tensor
        min_val = torch.min(X)
        max_val = torch.max(X)
        denom = max_val - min_val + eps

        X = (X - min_val) / denom
        X = torch.round(X)
        for idx in range(X.shape[0]):
            x_expanded = X[idx].unsqueeze(0)[0].tolist()  # Get the counts of each distinct value in the tensor
            counts = {value: x_expanded.count(value) for value in
                      x_expanded}  # Count the number of occurences of each unique value in x_expanded
            probs = [count / len(x_expanded) for count in
                     counts.values()]  # Calculate the probability of each unique value
            y.append((1 - sum([prob ** q for prob in probs])) / (
                        q - 1 + eps))  # for each row in X_train, calculate the Tsalis Entropy and store in y_train
        y = torch.tensor(y).reshape(num_samples)
        

    elif fin_type == "entropy_q":
        q = 1.1
        eps = 1e-8
        y = []
        X = torch.rand(num_samples, dim)

        # Normalize the input using min-max scaling across all values in the tensor
        min_val = torch.min(X)
        max_val = torch.max(X)
        denom = max_val - min_val + eps

        X = (X - min_val) / denom
        X = torch.round(X )
        for idx in range(X.shape[0]):
            x_expanded = X[idx].unsqueeze(0)[0].tolist()  # Get the counts of each distinct value in the tensor
            counts = {value: x_expanded.count(value) for value in
                      x_expanded}  # Count the number of occurences of each unique value in x_expanded
            probs = [count / len(x_expanded) for count in
                     counts.values()]  # Calculate the probability of each unique value
            y.append((1 - sum([prob ** q for prob in probs])) / (q - 1 + eps))  # for each row in X_train, calculate the Tsalis Entropy and store in y_train
        y = torch.tensor(y).reshape(num_samples)

    elif fin_type == "count_0":
        X = torch.rand(num_samples, dim)
        nbins = 10
        n_samples, n_features = X.shape
        y = torch.zeros(n_samples)

        for i in range(n_samples):
            # Discretize x into bins
            x_discrete = (X[i, :] * (nbins - 1)).round().int()
            # Estimate the probability distribution of x
            counts = torch.bincount(x_discrete, minlength=nbins)
            y[i] = (counts[0])
        y=y.unsqueeze(1)

    elif fin_type == "count":
        X = torch.rand(num_samples, dim)
        nbins = 10
        n_samples, n_features = X.shape
        y = torch.zeros(n_samples,10)

        for i in range(n_samples):
            # Discretize x into bins
            x_discrete = (X[i, :] * (nbins - 1)).round().int()
            # Estimate the probability distribution of x
            counts = torch.bincount(x_discrete, minlength=nbins)
            y[i] = (counts)
        y=y.unsqueeze(1)

    return X, y


# Define a function to estaimte shanon entropy given a numpy array
def shanon_entropy(x):
    # round numpy array to 2 decimal places
    x = np.round(x, 1)

    # Count the number of occurences of each unique value in the array
    counts = collections.Counter(x)

    # Calculate the probability of each unique value
    probs = [count / len(x) for count in counts.values()]
    # Calculate the shanon entropy
    entropy = -sum([prob * math.log(prob, 2) for prob in probs])
    return entropy





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fin_type", type=str, default="shanon_entropy",choices=["count_0","mean", "std", "shanon_entropy", "max", "min", "kurtosis", "entropy_q", "tsallis_entropy"])
    parser.add_argument("--batch_size", type=int, default=30000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1E-4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--train_data_size", type=int, default=2000000)
    parser.add_argument("--validate_data_size", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=1024)

    args = parser.parse_args()

    args.loss_function= nn.L1Loss()

    args=vars(args)

    # Set random seed for reproducibility
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])


    if args["fin_type"] == "mean":
        FIN_model = MeanFIN(args["dim"])
    elif args["fin_type"] == "std":
        mean_net = MeanFIN(args["dim"])
        mean_net.load_state_dict(torch.load(FIN_path + trained_fin_path["mean_{}".format(args["dim"])]))
        FIN_model = STDFIN(args["dim"],mean_net)
    elif args["fin_type"] == "shanon_entropy":
        FIN_model = EntropyFIN(args["dim"])
    elif args["fin_type"] == "max":
        FIN_model = MaxFIN(args["dim"])
    elif args["fin_type"] == "min":
        FIN_model = MinFIN(args["dim"])
    elif args["fin_type"] == "kurtosis":
        mean_net = MeanFIN(args["dim"])
        mean_net.load_state_dict(torch.load(FIN_path + trained_fin_path["mean_{}".format(args["dim"])]))
        std_net = STDFIN(args["dim"],mean_net)
        std_net.load_state_dict(torch.load(FIN_path + trained_fin_path["std_{}".format(args["dim"])]))
        FIN_model = KurtosisFIN(args["dim"],mean_net,std_net)

    elif args["fin_type"] == "entropy_q":
        FIN_model = EntropyFINQ(args["dim"])
    elif args["fin_type"] == "tsallis_entropy":
        FIN_model = TsallisEntropyFIN(args["dim"])
    elif args["fin_type"] == "count_0":
        FIN_model = BaseFIN(args["dim"])
    elif args["fin_type"] == "count":
        FIN_model = CountFIN(args["dim"])
    else:
        raise ValueError("fin_type must be mean, shanon_entropy or max")


    def worker_init_fn():
        random.seed(args["seed"])

    if args ["n_gpu"] > 1:
        FIN_model = nn.DataParallel(FIN_model)


    # Generate random data
    train_X, train_y = create_random_data(args["train_data_size"], args["dim"], fin_type=args["fin_type"])
    train_dataset = FINDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True,
                                  worker_init_fn=worker_init_fn)

    val_X, val_y = create_random_data(args["validate_data_size"], args["dim"], fin_type=args["fin_type"])
    val_dataset = FINDataset(val_X, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,
                                worker_init_fn=worker_init_fn)

    test_X, test_y = create_random_data(args["validate_data_size"], args["dim"], fin_type=args["fin_type"])
    test_dataset = FINDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,
                                 worker_init_fn=worker_init_fn)

    FIN_model, losses = train_model(FIN_model, train_dataloader, val_dataloader, args)
    loss, test_output = eval_model(FIN_model, test_dataloader, args["loss_function"])
    print(f"Test Loss: {loss:.8f}")

    test_y = test_y.squeeze().detach().numpy().tolist()

    y_hat=np.array(test_output).round()
    y=np.array(test_y)

    print("mean absolute percentage error",abs(((y - y_hat) / y)).mean() * 100)

    plt.scatter(test_output, test_y, marker='o')

    # plot the ideal line of y=x with a range based on the min and max values of y_train
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], 'k--', lw=2)

    # add a title and axis labels
    plt.title('Value Approximation {} {}'.format(args["fin_type"],np.round(loss.item(), 6)))
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')

    # add a legned
    plt.legend(['Predictions', 'Ideal Performance'])
    if not (os.path.exists(FIN_path+"/FIN-plots/{}".format(args["fin_type"]))):
        os.makedirs(FIN_path+"/FIN-plots/{}".format(args["fin_type"]))
    plt.savefig(FIN_path+"/FIN-plots/{}/test_{}_{}.png".format(args["fin_type"],args["fin_type"], np.round(loss.item(), 6)))
    plt.show()

    # save model
    save_path = FIN_path+"/fin_modules/{}/{}_{}_n_{}.pth".format(args["fin_type"],args["fin_type"], np.round(loss.item(), 6),args["dim"])
    print("save path: ", save_path)
    if isinstance(FIN_model, nn.DataParallel):
        FIN_model = FIN_model.module
    if not (os.path.exists(FIN_path+"/fin_modules/{}".format(args["fin_type"]))):
        os.makedirs(FIN_path+"/fin_modules/{}".format(args["fin_type"]))
    torch.save(FIN_model.state_dict(), save_path)