import torch
import torch.nn as nn
import torch.optim as optim
import Entropy_FIN as ef
import argparse
from tqdm import tqdm
import Trainer as trainer
from torch.optim import Adam
class RangeClassifer(nn.Module):
    def __init__(self):
        super(RangeClassifer, self).__init__()
        self.fc1 = nn.Linear(3, 10) # 3 inputs (a, b, x) to 10 hidden nodes
        self.fc2 = nn.Linear(10, 1) # 10 hidden nodes to 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Sigmoid for binary classification
        return x



def check_range(x):



    return (x[:, 0] >= x[:, 1]) & (x[:, 0] <= x[:, 2]).long()


train_X, test_X ,validate_X = torch.rand(1000000,3), torch.rand(10000,3), torch.rand(1000,3)

train_X[:, 1:3] = torch.sort(train_X[:, 1:3], dim=1)[0]
test_X[:, 1:3] = torch.sort(test_X[:, 1:3], dim=1)[0]
validate_X[:, 1:3] = torch.sort(validate_X[:, 1:3], dim=1)[0]

train_y, test_y ,validate_y = check_range(train_X).unsqueeze(1), check_range(test_X).unsqueeze(1), check_range(validate_X).unsqueeze(1)

tarin_dataset = ef.FINDataset   (train_X, train_y)
test_dataset = ef.FINDataset   (test_X, test_y)
validate_dataset = ef.FINDataset   (validate_X, validate_y)

train_dataloader = torch.utils.data.DataLoader(tarin_dataset, batch_size=5000, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5000, shuffle=False)
validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=1000, shuffle=False)



# Create the model instance

# load model
model_path= 'outputs/fold1_best_model.pth'
state_dict = torch.load(model_path)
model = RangeClassifer()
model.load_state_dict(state_dict)



args = argparse.Namespace()
args.max_epochs=30
args.lr=0.001
args.patience=10
args.loss_function= nn.BCELoss(reduction="sum")
args.k_fold=1
args.task="classification"
args.l2_decay=0.001
args.optimizer=Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
args.output_root = 'outputs'
args.criterion_root= args.output_root +'/criterion'
args.plot_root= args.output_root +'/plot'

data=[{"train":train_dataloader,"test":test_dataloader,"validate":validate_dataloader}]



Trainer =trainer.Trainer(model, data, args)
acc,loss,_,_ = Trainer.evaluate(test_dataloader)
print(acc,loss)