import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import random
import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os
import tqdm
import argparse


def plot_Matrix(cfg, y, yp):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp)
    np.save(os.path.join(cfg.criterion_root, f'confusion_matrix.npy'), cm)
    cm = cm.astype('float32')
    for i in range(len(cm)):
        cm[i] /= np.sum(cm[i])
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(f'{cm[x, y]:.3f}', xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(cfg.plot_root, f'confusion_matrix.jpg'))

class Trainer:
    def __init__(self,model, folds_data :list , cfg):
        super().__init__()

        self.cfg = cfg
        self.all_data = folds_data


        self.init_params =copy.deepcopy(model.state_dict())


        self.training_accs, self.training_losses = [], []
        self.val_accs, self.val_losses = [], []
        self.y, self.y_pred = [], []
        self.best_models = []

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.model.to(self.device)
        self.optimizer = cfg.optimizer
        # self.model.show_parameter_num()

        output_root = self.cfg.output_root
        # assert not os.path.exists(output_root), 'output_root already exists, files would be overwrited.'
        # remove output_root
        # if os.path.exists(output_root):
        #     os.system(f'rm -rf {output_root}')
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        if not os.path.exists(self.cfg.criterion_root):
            os.mkdir(self.cfg.criterion_root)
        if not os.path.exists(self.cfg.plot_root):
            os.mkdir(self.cfg.plot_root)

    def run(self, train_dataloader, validation_dataloader , test_dataloader , fold, max_epochs):
        fold_training_accs, fold_training_losses = [], []
        fold_test_accs, fold_test_losses = [], []
        fold_val_accs, fold_val_losses = [], []
        fold_ys, fold_y_preds = [], []
        tqdm_epochs = tqdm.tqdm(range(1, max_epochs + 1))
        best_loss = np.inf


        for epoch in tqdm_epochs:
            tqdm_epochs.set_description(f'epoch{epoch}')
            # print(f'epoch{epoch}...')
            t_start = time.perf_counter()

            fold_training_acc, fold_training_loss = self.train(train_dataloader)
            fold_val_acc, fold_val_loss, _, _ = self.evaluate(validation_dataloader)
            fold_test_acc, fold_test_loss, fold_y, fold_y_pred = self.evaluate(test_dataloader)
            t_end = time.perf_counter()

            tqdm_epochs.set_postfix(time=np.round((t_end - t_start), 2), acc=fold_training_acc, loss=fold_training_loss,
                                    val_acc=fold_val_acc, val_loss=fold_val_loss
                                    )
            if fold_val_loss < best_loss:
                best_loss = fold_val_loss
                best_model = copy.deepcopy(self.model)

            fold_training_accs.append(fold_training_acc)
            fold_training_losses.append(fold_training_loss)
            fold_val_accs.append(fold_val_acc)
            fold_val_losses.append(fold_val_loss)
            fold_test_accs.append(fold_test_acc)
            fold_test_losses.append(fold_test_loss)
            fold_ys.append(fold_y)
            fold_y_preds.append(fold_y_pred)

        self.training_accs.append(fold_training_accs)
        self.training_losses.append(fold_training_losses)
        self.val_accs.append(fold_val_accs)
        self.val_losses.append(fold_val_losses)
        best_idx = torch.tensor(fold_val_losses).min(0)[1].item()
        self.y += fold_ys[best_idx]
        self.y_pred += fold_y_preds[best_idx]
        self.best_models.append(copy.deepcopy(best_model.state_dict))

        names = ['acc', 'loss', 'val_acc', 'val_loss']
        values = [fold_training_accs, fold_training_losses, fold_val_accs, fold_val_losses]
        for name, value in zip(names, values):
            with open(os.path.join(self.cfg.criterion_root, f'fold{fold + 1}_{name}'), 'wb') as f:
                pickle.dump(value, f)
        # save best model
        torch.save(best_model.state_dict(), os.path.join(self.cfg.output_root, f'fold{fold + 1}_best_model.pth'))


    def train(self, train_dataloader):
        self.model.train()

        training_acc, training_loss = 0, 0
        n_X = 0
        for batch_i, data in enumerate(train_dataloader):
            X,y= data
            X=X.to(self.device)
            y=y.to(self.device)
            n_X+=X.shape[0]
            self.optimizer.zero_grad()
            out = self.model(X)
            if self.cfg.task=="classification":
                acc = out.round().squeeze().eq(y.squeeze()).sum().item()
            else:
                acc = 0
            loss = self.cfg.loss_function(out, y)
            loss_num = loss.detach().item()
            training_acc += acc
            training_loss += loss_num

            loss.backward()
            self.optimizer.step()

        training_acc /= n_X
        training_loss /= n_X

        return training_acc, training_loss

    def evaluate(self, val_dataloader):
        self.model.eval()

        all_acc = 0
        all_loss = 0
        batch_y, batch_y_pred = [], []

        n_X = 0
        for batch,data in enumerate(val_dataloader):
            X,y= data
            n_X+=X.shape[0]
            with torch.no_grad():
                X=X.to(self.device)
                y=y.to(self.device)
                logits = self.model(X)
                if self.cfg.task=="classification":
                    pred = logits.round().squeeze()
                else:
                    pred = logits


            loss = self.cfg.loss_function(logits, y)
            if self.cfg.task=="classification":
                all_acc += pred.eq(y.squeeze()).sum().item()
            all_loss += loss.detach().item()
            batch_y += y.squeeze().tolist()
            batch_y_pred += pred.squeeze().tolist()

        epoch_acc = all_acc / n_X
        epoch_loss = all_loss / n_X

        return epoch_acc, epoch_loss, batch_y, batch_y_pred

    def save_overall_figures(self):
        training_accs = np.mean(self.training_accs, axis=0)
        training_losses = np.mean(self.training_losses, axis=0)
        val_accs = np.mean(self.val_accs, axis=0)
        val_losses = np.mean(self.val_losses, axis=0)

        epochs = range(1, len(training_losses) + 1)
        plt.plot(epochs, training_accs, 'bo', label='Training acc')
        plt.plot(epochs, val_accs, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.plot_root, f'acc.jpg'))

        plt.figure()
        plt.plot(epochs, training_losses, 'bo', label='Training loss')
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.plot_root, f'loss.jpg'))

        plt.figure()
        plot_Matrix(self.cfg, self.y, self.y_pred)

        precision_recall_f1 = precision_recall_fscore_support(self.y, self.y_pred, average='macro')
        precision_recall_f1_file = open(os.path.join(self.cfg.criterion_root, f'precision_recall_f1'), 'wb')
        pickle.dump(precision_recall_f1, precision_recall_f1_file)
        precision_recall_f1_file.close()

        y_file = open(os.path.join(self.cfg.criterion_root, f'y'), 'wb')
        y_pred_file = open(os.path.join(self.cfg.criterion_root, f'y_pred'), 'wb')
        pickle.dump(self.y, y_file)
        pickle.dump(self.y_pred, y_pred_file)
        y_file.close()
        y_pred_file.close()

        print("-" * 10, "Overal Performance", "-" * 10)
        print("acc: {:.3f}".format(accuracy_score(self.y, self.y_pred)))
        print(
            f'precision: {precision_recall_f1[0]:.3f}\trecall: {precision_recall_f1[1]:.3f}\tf1: {precision_recall_f1[2]:.3f}')

    def start(self):
        for i in range(self.cfg.k_fold):
            print(f'=====fold {i + 1}=====')
            # self.model.reset_parameters()
            # reset model parameters
            self.model.load_state_dict(self.init_params)

            train_dataloader, test_dataloader,validate_dataloader = self.all_data[i]["train"], self.all_data[i]["test"], self.all_data[i]["validate"]
            self.run(train_dataloader, test_dataloader, validate_dataloader ,fold=i, max_epochs=self.cfg.max_epochs)

            del train_dataloader, test_dataloader ,validate_dataloader
            torch.cuda.empty_cache()

        self.save_overall_figures()


def set_seed(seed):
    """Set all seeds to make results reproducible (deterministic mode)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True  # Necessary for some CuDNN algorithms.
    # torch.backends.cudnn.benchmark = False


if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument("--n_gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fin_type", type=str, default="shanon_entropy",choices=["shanon_entropy", "shanon_entropy_dynamic_bin"])
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1E-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--train_data_size", type=int, default=1000000)
    parser.add_argument("--validate_data_size", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--bin_num", type=int, default=10)

    args = parser.parse_args()

    args.loss_function= nn.L1Loss(reduction="mean" )


    def my_app(cfg: args):
        # Set random seed for reproducibility
        set_seed(cfg.seed)
        trainer = Trainer(cfg)
        trainer.start()


    my_app()

