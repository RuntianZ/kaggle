import numpy as np 
import torch
import torch.nn.functional as F
import polars as pl 
import pandas as pd 
from torch.utils.data import IterableDataset, DataLoader
import utils
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dims: str, activation: str = "relu"):
        super().__init__()
        if hidden_dims:
            hidden_sizes = hidden_dims.split(",")
            self.net = torch.nn.Sequential()
            in_size = input_size
            for i, h in enumerate(hidden_sizes):
                h_size = int(h)
                self.net.add_module(f"linear_{i}", torch.nn.Linear(in_size, h_size))
                if activation == "relu":
                    self.net.add_module(f"relu_{i}", torch.nn.ReLU())
                elif activation == "tanh":
                    self.net.add_module(f"tanh_{i}", torch.nn.Tanh())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                in_size = h_size
            self.net.add_module("output", torch.nn.Linear(in_size, output_size))
        else:
            # linear model
            self.net = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.net(x)
        return out


class SeparateMLP(torch.nn.Module):
    # Separate MLP for each output dimension
    def __init__(self, input_size: int, output_size: int, hidden_dims: str, activation: str = "relu"):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for _ in range(output_size):
            model = MLP(input_size=input_size, output_size=1, hidden_dims=hidden_dims, activation=activation)
            self.models.append(model)

    def forward(self, x):
        # Forward pass for each model and concatenate the outputs
        outputs = [model(x) for model in self.models]
        return torch.cat(outputs, dim=1)


def evaluate(loader, model, device, loss_func):
    model.eval()
    y_all = None
    w_all = None
    out_all = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"]
            w = batch["w"]
            out = model(x)
            if y_all is None:
                y_all = y.cpu()
                w_all = w.cpu()
                out_all = out.cpu()
            else:
                y_all = torch.cat([y_all, y.cpu()], dim=0)
                w_all = torch.cat([w_all, w.cpu()], dim=0)
                out_all = torch.cat([out_all, out.cpu()], dim=0)
            
        if loss_func == "mse":
            # weighted mse, weighted r2, univariate slope, yhat std, slope_sd
            res = F.mse_loss(out_all, y_all, reduction="none") * w_all
            wmse = res.mean().item()
            tot = (y_all ** 2 * w_all)
            r2 = 1 - res.sum().item() / tot.sum().item()
            utils.JLOG(f"WMSE: {wmse}, WR2: {r2}")
            return {
                "wmse": wmse,
                "wr2": r2,
            }
        elif loss_func == "binary":
            y_all = y_all.view(-1)
            out_all = out_all.view(-1)
            # accuracy for pos and neg classes, auc, f1
            y_true = (y_all > 0.5).float()
            y_prob = torch.sigmoid(out_all)
            y_pred = (y_prob > 0.5).float()
            tp = ((y_true == 1) & (y_pred == 1)).float()
            tn = ((y_true == 0) & (y_pred == 0)).float()
            fp = ((y_true == 0) & (y_pred == 1)).float()
            fn = ((y_true == 1) & (y_pred == 0)).float()
            acc_pos = (tp.sum() / (tp.sum() + fn.sum())).item()
            acc_neg = (tn.sum() / (tn.sum() + fp.sum())).item()
            auc_roc = roc_auc_score(y_true.numpy(), y_prob.numpy())
            f1 = f1_score(y_true.numpy(), y_pred.numpy())
            utils.JLOG(f"Acc_pos: {acc_pos} ({tp.sum().item()} / {tp.sum().item() + fn.sum().item()}), Acc_neg: {acc_neg} ({tn.sum().item()} / {tn.sum().item() + fp.sum().item()}), AUC_ROC: {auc_roc}, F1: {f1}")
            return {
                "acc_pos": acc_pos,
                "acc_neg": acc_neg,
                "auc_roc": auc_roc,
                "f1": f1,
            }
    
    return {}


def train_mlp(ds_train, ds_test, device, hidden_dims, activation, separate: bool,
              batch_size, epochs, lr, input_size, output_size, loss_func: str,
              test_on_train: bool, result_file: str, log_file: str):
    train_loader = DataLoader(ds_train, batch_size=batch_size)
    test_loader = DataLoader(ds_test, batch_size=batch_size)

    if separate:
        model = SeparateMLP(input_size=input_size, output_size=output_size, hidden_dims=hidden_dims, activation=activation)
    else:
        model = MLP(input_size=input_size, output_size=output_size, hidden_dims=hidden_dims, activation=activation)
    model = model.to(device)

    if loss_func == "mse":
        criterion = torch.nn.MSELoss(reduction="none")
    elif loss_func == "binary":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss function: {loss_func}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    with open(log_file, "w") as f:
        f.write("step,loss\n")
    global_step = 0
    all_results = {"epoch": []}
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            w = batch["w"].to(device).float()

            optimizer.zero_grad()
            outputs = model(x).float()

            if loss_func == "binary":
                outputs = outputs.view(-1)
                y = y.view(-1)

            loss = criterion(outputs, y)
            loss = (loss * w).mean()
            loss_t = loss.item()
            loss.backward()
            optimizer.step()
            total_loss += loss_t * x.size(0)
            total_samples += x.size(0)
            with open(log_file, "a") as f:
                f.write(f"{global_step},{loss_t}\n")
            global_step += 1
        avg_loss = total_loss / total_samples
        utils.JLOG(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        # scheduler.step()

        if test_on_train:
            utils.JLOG(f"Epoch {epoch+1} evaluation on training set:")
            evaluate(train_loader, model, device, loss_func)
        
        utils.JLOG(f"Epoch {epoch+1} evaluation on test set:")
        results = evaluate(test_loader, model, device, loss_func)
        for k, v in results.items():
            if k not in all_results:
                all_results[k] = []
            all_results[k].append(v)
        all_results["epoch"].append(epoch + 1)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(result_file, index=False)
    utils.JLOG(f"Training complete. Results saved to {result_file}")


class MyDataset(IterableDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        self.x = x
        self.y = y
        self.w = w

    def __iter__(self):
        for i in range(self.x.size(0)):
            yield {
                "x": self.x[i],
                "y": self.y[i],
                "w": self.w[i],
            }


if __name__ == "__main__":
    # sample dataset
    x_train = torch.randn(1000, 10)
    # y_train = (x_train[:, 0] + x_train[:, 1] > 0).long()
    y_train = (torch.randn(1000, 1) > 0).long()
    # print(y_train.float().sum())
    w_train = torch.ones(1000).float()

    ds_train = MyDataset(x_train, y_train, w_train)

    x_test = torch.randn(400, 10)
    # y_test = (x_test[:, 0] + x_test[:, 1] > 0).long()
    y_test = (torch.randn(400, 1) > 0).long()
    w_test = torch.ones(400).float()
    ds_test = MyDataset(x_test, y_test, w_test)

    train_mlp(ds_train=ds_train, ds_test=ds_test, device="cpu", hidden_dims="20,10", activation="relu", separate=False,
                batch_size=16, epochs=150, lr=0.01, input_size=10, output_size=1, loss_func="binary",
                test_on_train=True, result_file="mlp_results.csv", log_file="mlp_log.csv")