import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import random
import numpy as np


from tqdm import tqdm as tqdm


import os

FIN_path = os.path.dirname(__file__)
torch.autograd.set_detect_anomaly(True)


class RangeClassifer(nn.Module):
    def __init__(self):
        super(RangeClassifer, self).__init__()
        self.fc1 = nn.Linear(3, 10) # 3 inputs (a, b, x) to 10 hidden nodes
        self.fc2 = nn.Linear(10, 1) # 10 hidden nodes to 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Sigmoid for binary classification
        return x

def smooth_transition(x, a, b):
    """
        Parameters:
        x (float or array-like): The input value or array of values for which the smooth transition is computed.
        a (float): The start of the transition range. For x <= a, the function will approach 0.
        b (float): The end of the transition range. For x >= b, the function will approach 0.

        Returns:
        float or array-like: The output of the smooth transition function, which will be close to 1
        when x is between a and b and will smoothly transition towards 0 outside this range.
        """
    # Defining a sigmoid function

    sigmoid = lambda y: 1 / (1 + torch.exp(-y+1e-10))

    # Scale factors to control the sharpness of the transition
    k1, k2 = 80,80
    # x=x.clip((a - 2 * a), (b + 2 * b))
    # Applying the sigmoid function to create a smooth transition
    return sigmoid(k1 * (x - a)) * (1 - sigmoid(k2 * (x - b)))


def range_classifier(model,x,a,b):
    n,m = x.shape
    x= x.reshape(n*m,1)
    all_bins_counts=[]
    for i in range(a.shape[0]):
        model_input= torch.cat([x,torch.ones((n*m,1)).to(x.device)*a[i],torch.ones((n*m,1)).to(x.device)*b[i]],dim=1)
        y_bin = model(model_input)
        y_bin = y_bin.reshape(n,m,1)
        all_bins_counts.append(y_bin)
    y_hat= torch.cat(all_bins_counts, dim=2)
    return y_hat





class EntropyBinningWidthModel(nn.Module):
    def __init__(self, initial_nbins=10):
        super(EntropyBinningWidthModel, self).__init__()
        self.initial_nbins = initial_nbins
        self.raw_bin_widths = nn.Parameter(torch.rand(self.initial_nbins))  # Raw parameters for bin widths

    def forward(self, X):
        device= X.device
        # Apply softmax to raw bin widths to get positive values that sum to 1
        normalized_bin_widths = torch.nn.functional.softmax(self.raw_bin_widths, dim=0)

        # Compute bin edges from normalized bin widths
        bin_edges = torch.cumsum(normalized_bin_widths, dim=0)
        bin_edges = torch.cat([torch.tensor([0]).to(device), bin_edges])  # Add starting edge


        # sorted_bin_edges = self.merge_bins(sorted_bin_edges)
        X_expanded = X.unsqueeze(2)  # Shape: [n_samples, n_features, 1]
        bins_expanded = bin_edges.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, nbins + 1]

        # Vectorized computation of bin counts
        bin_counts = smooth_transition(X_expanded, bins_expanded[:, :, :-1], bins_expanded[:, :, 1:]).sum(dim=1)

        p_bins = bin_counts / X.shape[1]

        # Compute entropy
        y = -torch.sum(p_bins * torch.log2(p_bins + 1e-10), dim=1)

        return y.unsqueeze(-1)




class EntropyDynamicBinningModel(nn.Module):
    def __init__(self, initial_nbins=10):
        super(EntropyDynamicBinningModel, self).__init__()
        self.initial_nbins = initial_nbins
        self.bin_edges = nn.Parameter(torch.rand(self.initial_nbins+1))
        self.alpha = nn.Parameter(torch.tensor([0.1]))
    def merge_bins(self,sorted_bin_edges):
        # Calculate bin widths
        bin_widths = sorted_bin_edges[1:] - sorted_bin_edges[:-1]

        # Identify bins to merge
        small_bins = bin_widths < self.alpha
        indices_to_remove=torch.where(small_bins)[0]
        mask = torch.ones(sorted_bin_edges.size(0), dtype=torch.bool)
        mask[indices_to_remove + 1] = False
        sorted_bin_edges = sorted_bin_edges[mask]
        return sorted_bin_edges

    def forward(self, X):
        sorted_bin_edges, _ = torch.sort(self.bin_edges)
        sorted_bin_edges = self.merge_bins(sorted_bin_edges)
        self.bin_edges=nn.Parameter(sorted_bin_edges)
        X_expanded = X.unsqueeze(2)  # Shape: [n_samples, n_features, 1]
        bins_expanded = sorted_bin_edges.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, nbins + 1]

        # Vectorized computation of bin counts
        bin_counts = smooth_transition(X_expanded, bins_expanded[:, :, :-1], bins_expanded[:, :, 1:]).sum(dim=1)

        p_bins = bin_counts / X.shape[1]

        # Compute entropy
        y = -torch.sum(p_bins * torch.log2(p_bins + 1e-10), dim=1)

        return y.unsqueeze(-1)



class EntropyBinningModel(nn.Module):
    def __init__(self, initial_nbins=10, range_classifier_path=None):
        super(EntropyBinningModel, self).__init__()
        self.initial_nbins = initial_nbins
        self.bin_edges = nn.Parameter(torch.rand(self.initial_nbins+1))
        self.range_classifier_path=range_classifier_path
        if (self.range_classifier_path is not None):
            self.range_classifier = RangeClassifer()
            self.range_classifier.load_state_dict(torch.load(range_classifier_path))
            # freeze weights
            for param in self.range_classifier.parameters():
                param.requires_grad = False

    def forward(self, X):
        sorted_bin_edges, _ = torch.sort(self.bin_edges)
        # for i in range(n_samples):
        #     x_expanded = X[i, :].unsqueeze(0)
        #     for b in range(self.initial_nbins):
        #         bin_count = smooth_transition(x_expanded, sorted_bin_edges[b], sorted_bin_edges[b + 1]).sum()
        #         p_bin = bin_count / n_features
        #         y[i] += -p_bin * torch.log2(p_bin + 1e-10)
        # Expand dimensions for broadcasting
         # Shape: [1, 1, nbins + 1]
        if self.range_classifier_path is not None:
            bin_counts = range_classifier(self.range_classifier, X, sorted_bin_edges[:-1],sorted_bin_edges[ 1:]).sum(dim=1)
            p_bins = bin_counts / X.shape[1]
            y = -torch.sum(p_bins * torch.log2(p_bins + 1e-10), dim=1)
        else:
            X_expanded = X.unsqueeze(2)  # Shape: [n_samples, n_features, 1]
            bins_expanded = sorted_bin_edges.unsqueeze(0).unsqueeze(0)
            # Vectorized computation of bin counts
            bin_counts = smooth_transition(X_expanded, bins_expanded[:, :, :-1], bins_expanded[:, :, 1:]).sum(dim=1)
            p_bins = bin_counts / X.shape[1]

            # Compute entropy
            y = -torch.sum(p_bins * torch.log2(p_bins + 1e-10), dim=1)
        return y.unsqueeze(-1)



def train_model(model, train_dataloader, val_dataloader, hyperparameters):
    epochs = hyperparameters.epochs
    loss_function = hyperparameters.loss_function
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr, weight_decay=0.0001)
    patience = hyperparameters.patience
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience,
                                           verbose=True)  # Reduce learning rate when a metric has stopped improving.
    best_val_loss = float("inf")
    counter = 0
    losses = []
    finish_training=False
    grad_norms=[]

    pbar = tqdm(total=(epochs))
    for epoch in range(epochs):
        model = model.cuda()
        model.train()  # Denotes that we are in training mode
        epoch_loss = 0
        for batch, data in enumerate(train_dataloader):
            # if epoch==119:
            #     print("epoch 119")
            X, y = data
            optimizer.zero_grad()  # Clear out the gradients from the last `epoch`:
            yhat = model(X.cuda())  # Compute the output of the model, yhat, given xif torch.isnan(yhat).any():
            if torch.isnan(yhat).any():
                raise ValueError("NaN values detected in model output")


            loss = loss_function(yhat.cuda(), y.cuda())  # Compute the loss
            if torch.isnan(loss).any():
                raise ValueError("NaN values detected in loss")


            # print("Gradients before clipping:", model.bin_edges.grad)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print("Gradients after clipping:", model.bin_edges.grad)

            loss.backward()



            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # raise ValueError(f"NaN values detected in the gradients of {name}")

            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    grad_norm = parameter.grad.norm()
                    grad_norms.append(grad_norm)
            #         print(f"Gradient norm for {name}: {grad_norm}")


            optimizer.step()  # Adjust the paramter values in the direction of the gradients
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    raise ValueError(f"NaN values detected in the parameters of {name}")

            epoch_loss += loss.item()

        if finish_training:
            break
        if epoch % 50 == 0:
            print(model.module.bin_edges)
            print("-" * 50)
            if hyperparameters.fin_type == "shanon_entropy_dynamic_bin":
                # print(model.module.alpha)
                # print(model.module.merge_bins(torch.sort(model.module.bin_edges)[0]))
                print(torch.sort(model.module.raw_bin_widths)[0])
            print("-" * 50)
        losses.append(epoch_loss)  # Save the loss at each interation for plotting

        # Evaluate the model on the validation set
        val_loss, _ = eval_model(model, val_dataloader, loss_function)

        # Update learning rate with ReduceLROnPlateau scheduler
        scheduler_plateau.step(val_loss)

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


def create_random_data(num_samples, dim, fin_type="shanon_entropy", nbins=10):
    if fin_type == "shanon_entropy" or fin_type == "shanon_entropy_dynamic_bin":
        X = torch.rand(num_samples, dim)
        y = compute_entropy(X, nbins)


    elif fin_type == "shanon_entropy_dynamic_bin--":
        X = torch.rand(num_samples, dim)
        n_samples, n_features = X.shape
        y= torch.zeros((num_samples))

        for s in range(n_samples):
            nbins= get_num_bins(X[s].numpy())
            bin_edges = torch.linspace(0, 1, nbins + 1)
            # Initialize tensor to hold counts
            bin_counts = torch.zeros((nbins))
            # Count the number of features in each bin for each sample
            for i in range(nbins):
                # Define the interval
                left_edge = bin_edges[i]
                right_edge = bin_edges[i + 1]

                # Count features within the interval
                bin_counts[i] = ((X[s] > left_edge) & (X[s] <= right_edge)).sum()

            # Normalize the histogram to get probabilities
            p_bins = bin_counts / X.shape[1]
            # Compute entropy
            y[s] = -torch.sum(p_bins * torch.log2(p_bins + 1e-10))

    else:
        raise ValueError("fin_type must be shanon_entropy or shanon_entropy_dynamic_bin")

    return X,y.unsqueeze(1)


def compute_entropy(X, nbins=10):
    num_samples = X.shape[0]
    bin_edges = torch.linspace(0, 1, nbins + 1)
    # bin_edges= torch.tensor([0,0.1,0.9,1])
    # print("Actual Bin Edges:",bin_edges)
    # Initialize tensor to hold counts
    bin_counts = torch.zeros((num_samples, nbins))
    # Count the number of features in each bin for each sample
    for i in range(nbins):
        # Define the interval
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]

        # Count features within the interval
        bin_counts[:, i] = ((X > left_edge) & (X <= right_edge)).sum(dim=1)
    # Normalize the histogram to get probabilities
    p_bins = bin_counts / X.shape[1]
    # Compute entropy
    y = -torch.sum(p_bins * torch.log2(p_bins + 1e-10), dim=1)
    return y


def get_num_bins(data):

    # Calculating the number of bins using Freedman-Diaconis Rule
    IQR = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range
    N = len(data)  # Number of data points
    # Bin width as per Freedman-Diaconis Rule
    bin_width = 2 * IQR / (N ** (1 / 3))
    # Number of bins
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return num_bins






if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--n_gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fin_type", type=str, default="shanon_entropy",choices=["shanon_entropy", "shanon_entropy_dynamic_bin"])
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1E-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--train_data_size", type=int, default=100000)
    parser.add_argument("--validate_data_size", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--bin_num", type=int, default=10)
    parser.add_argument("--range_classifier_path", type=str, default="outputs/fold1_best_model.pth")

    args = parser.parse_args()

    args.loss_function= nn.L1Loss(reduction="mean" )



    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



    if args.fin_type == "shanon_entropy":
        FIN_model = EntropyBinningModel(initial_nbins=args.bin_num,range_classifier_path=args.range_classifier_path)
    elif args.fin_type == "shanon_entropy_dynamic_bin":
        # FIN_model = EntropyDynamicBinningModel(initial_nbins=10)
        FIN_model = EntropyBinningWidthModel(initial_nbins=10)

    # FIN_model = EntropyDynamicBinningModel(initial_nbins=10)




    def worker_init_fn():
        random.seed(args.seed)

    if args.n_gpu > 1:
        FIN_model = nn.DataParallel(FIN_model)






    # Generate random data
    train_X, train_y = create_random_data(args.train_data_size, args.dim, fin_type=args.fin_type)
    train_dataset = FINDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  worker_init_fn=worker_init_fn)

    val_X, val_y = create_random_data(args.validate_data_size, args.dim, fin_type=args.fin_type)
    val_dataset = FINDataset(val_X, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                worker_init_fn=worker_init_fn)

    test_X, test_y = create_random_data(args.validate_data_size, args.dim, fin_type=args.fin_type)
    test_dataset = FINDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 worker_init_fn=worker_init_fn)

    FIN_model, losses = train_model(FIN_model, train_dataloader, val_dataloader, args)
    loss, test_output = eval_model(FIN_model, test_dataloader, args.loss_function)
    print(f"Test Loss: {loss:.8f}")

    test_y = test_y.squeeze().detach().numpy().tolist()

    y_hat=np.array(test_output)
    y=np.array(test_y)

    print("mean absolute percentage error",abs(((y - y_hat) / y)).mean() * 100)

    plt.scatter(test_output, test_y, marker='o')

    # plot the ideal line of y=x with a range based on the min and max values of y_train
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], 'k--', lw=2)

    # add a title and axis labels
    plt.title('Value Approximation {} {}'.format(args.fin_type,np.round(loss.item(), 6)))
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')

    # add a legned
    plt.legend(['Predictions', 'Ideal Performance'])
    if not (os.path.exists(FIN_path+"/FIN-plots/{}".format(args.fin_type))):
        os.makedirs(FIN_path+"/FIN-plots/{}".format(args.fin_type))
    plt.savefig(FIN_path+"/FIN-plots/{}/test_{}_{}.png".format(args.fin_type,args.fin_type, np.round(loss.item(), 6)))
    plt.show()

    # save model
    if args.range_classifier_path is not None:
        save_path = FIN_path+"/fin_modules/{}/{}_{}_n_{}_bins{}_range_classifier_[].pth".format(args.fin_type,args.fin_type, np.round(loss.item(), 6),args.dim,args.bin_num,"with_range_classifier")
    else:
        save_path = FIN_path+"/fin_modules/{}/{}_{}_n_{}_bins{}.pth".format(args.fin_type,args.fin_type, np.round(loss.item(), 6),args.dim,args.bin_num)
    print("save path: ", save_path)
    if isinstance(FIN_model, nn.DataParallel):
        FIN_model = FIN_model.module
    if not (os.path.exists(FIN_path+"/fin_modules/{}".format(args.fin_type))):
        os.makedirs(FIN_path+"/fin_modules/{}".format(args.fin_type))
    torch.save(FIN_model.state_dict(), save_path)

    for name, param in FIN_model.named_parameters():
        print(f"{name}: {param}")

    print(torch.sort(FIN_model.bin_edges)[0])
    print("distances")
    print(torch.sort(FIN_model.bin_edges)[0][1:]-torch.sort(FIN_model.bin_edges)[0][:-1])