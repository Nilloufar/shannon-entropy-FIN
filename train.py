
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from networks.FIN.Entropy_FIN import EntropyBinningModel, EntropyDynamicBinningModel



torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument("--n_gpu", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fin_type", type=str, default="shanon_entropy"
                    ,choices=["shanon_entropy", "shanon_entropy_dynamic_bin"])
parser.add_argument("--batch_size", type=int, default=50000)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--lr", type=float, default=1E-3)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--train_data_size", type=int, default=1000000)
parser.add_argument("--validate_data_size", type=int, default=5000)
parser.add_argument("--dim", type=int, default=400)

args = parser.parse_args()

args.loss_function= nn.L1Loss(reduction="mean" )



# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



if args.fin_type == "shanon_entropy":
    FIN_model = EntropyBinningModel(initial_nbins=10)
elif args.fin_type == "shanon_entropy_dynamic_bin":
    FIN_model = EntropyDynamicBinningModel(initial_nbins=3)

FIN_model = EntropyDynamicBinningModel(initial_nbins=3)




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

y_ha t =np.array(test_output) y=np.a rray(test_y)

print("mean absolute percentage error",abs( ((y - y_hat) / y)).mean() * 100)

plt.scatter(test_output, test_y, marker='o')

# plot the ideal line of y=x with a range based on the min and max values of y_train
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], 'k--', lw=2)

# add a title and axis labels
plt.title('Value Approximation {} {}'.format(args.fin_type,np.r ound(loss.item(), 6)))
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

# add a legned
plt.legend(['Predictions', 'Ideal Performance'])
if not (os.path.exists(FIN_path+"/F I N-plots/{}".format(args.fin_type))):
    os.makedirs(FIN_path+"/F I N-plots/{}".format(args.fin_type))
plt.savefig(FIN_
    path+"/F I N-plots/{}/test_{}_{}.png".format(args.fin_type,args .fin_type, np.round(loss.item(), 6)))
plt.show()

# save model
save_path = FIN_path+"/f i n_modules/{}/{}_{}_n_{}.pth".format(args.fin_type,args .fin_type, np.
                                                               round(loss.item(), 6),arg s.dim)
print("save path: ", save_path)
if isinstance(FIN_model, nn.DataParallel):
    FIN_model = FIN_model.module
if not (os.path.exists(FIN_path+"/ f in_modules/{}".format(args.fin_type))):
    os.makedirs(FIN_path+"/ f in_modules/{}".format(args.fin_type))
torch.save(FIN_model.state_dict(), save_path)

for name, param in FIN_model.named_parameters():
    print(f"{name}: {param}")

print(torch.sort(FIN_model.bin_edges)[0])
print("distances")
print(torch.sort(FIN_model.bin_edges)[0][1:]-to r ch.sort(FIN_model.bin_edges)[0][:-1])