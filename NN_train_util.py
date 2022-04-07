# File with util functions for training (on colab)
from NNClasses import *
from NNDataLoader import *
from torch.utils.data import DataLoader, SequentialSampler
import os
import io
import pickle
from matplotlib import cm
from prettytable import PrettyTable

# https://github.com/pytorch/pytorch/issues/16797
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def get_filename(L, trainSet=True, pathToThesis=""):
    f = os.path.join(pathToThesis, "data",
                     "train" if trainSet else "test", f"dataset_julia_L={L}")
    if os.path.isfile(f):
        return f
    else:
        f += "_cpu"
        if os.path.isfile(f):
            return f
        else:
            raise FileNotFoundError(f)


def load_data_from_pickle(L, Tc0=None, pathToThesis="", trainSet=True, max_data_size=0, train_on_vortex=False, normalize=False, device="cuda"):
    """loads dataset, can convert to train on vortices if needed"""
    f = get_filename(L, trainSet, pathToThesis)
    with open(f, "rb") as file:
        if device == "cuda":
            data: CustomDataset = pickle.load(file)
        else: # custom load if on cpu
            data: CustomDataset = CPU_Unpickler(file).load()
            data.device = "cpu"


    if Tc0:
        data.Tc = Tc0

    # by default: trainSet not normalized, testSet is
    data.normalized_data = not trainSet

    data.configs = data.configs.float()
    data.labels = data.labels.float()
    data.Ts = data.Ts.float()

    if data.max_size is None:
        data.max_size = len(data)
    if max_data_size != 0:
        data.shortenData(max_data_size)
    if train_on_vortex:
        data.spin_to_vortex()
        data.normalized_data = True

    if trainSet and normalize and (not train_on_vortex):
        data.normalize_data()
    elif (not trainSet) and (not data.normalized_data) and (not train_on_vortex):
        # test set is normalized by default
        data.configs *= 2*np.pi
    return data


def load_dataSet(root, L, Tc0, max_data_Size, train_on_vortex):
    """load train or test set
    """
    train_data = CustomDataset(root, L, Tc0, data_Size=max_data_Size)

    if train_on_vortex:
        train_data.spin_to_vortex()

    print(f"Training on {train_data.state}")
    print(f"Train set size: {len(train_data)} elements")

    return train_data


def plot_learning_curve(ax, loss_and_acc_scores, title=""):
    """Plots learning curve using the given scores.
    Note that these score are for one fold.
    """
    # {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    train_loss, val_loss, train_acc, val_acc = loss_and_acc_scores["train_loss"], loss_and_acc_scores[
        "val_loss"], loss_and_acc_scores["train_acc"], loss_and_acc_scores["val_acc"]

    # plot losses
    ax.plot(range(len(val_loss)), val_loss, '-', label=f"validation loss")
    ax.plot(range(len(train_loss)), train_loss, '-', label=f"train loss")

    # plot acc
    ax.plot(range(len(val_acc)), val_acc, '-', label=f"validation acc")
    ax.plot(range(len(train_acc)), train_acc, '-', label=f"train acc")
    ax.set_ylim(-.1, 1.1)
    ax.set_xlabel(f"Epoch")
    ax.set_ylabel(f"Loss / Accuracy")
    ax.legend()
    if title:
        ax.set_title(title)


def plot_learning_curve_L(ax, loss_and_acc_scores, L, title=""):
    """Plots learning curve using the given scores.
    Note that these score are for one fold.
    """
    # {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    _, val_loss, _, val_acc = loss_and_acc_scores["train_loss"], loss_and_acc_scores[
        "val_loss"], loss_and_acc_scores["train_acc"], loss_and_acc_scores["val_acc"]

    # colorindex
    c_idx = L/64

    # plot losses
    ax.plot(range(len(val_loss)), val_loss, '--', marker=L //
            8, color=cm.Blues(c_idx), label=f"loss L={L}")

    # plot acc
    ax.plot(range(len(val_acc)), val_acc, '-', marker=L //
            8, color=cm.Blues(c_idx), label=f"acc L={L}")
    ax.set_ylim(-.1, 1.1)
    ax.set_xlabel(f"Epoch")
    ax.set_ylabel(f"Loss / Accuracy")
    ax.legend()
    if title:
        ax.set_title(title)


def init_fig(ax):
    # decoration
    ax.set_xlim(-.1, 1.1)
    ax.set_ylabel("Sample amount")


def plot_hist(ax, labels, outputs, i, num_bins):
    # update labels
    ax.set_xlabel(f"Prediction at epoch {i+1}")
    labels_hist = ax.hist(labels[i], bins=num_bins, range=(
        0, 1), color=cm.Oranges(0.7), alpha=1)

    # update predictions
    predictions_hist = ax.hist(outputs[i], bins=num_bins, range=(
        0, 1), color=cm.Blues(.7), alpha=1)


def count_parameters(model):
    """Counts parameters of a model.
    Prints out in a nice table.
    Counts trainable model parameters."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        #if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def count_parameters_simple(model):
    """Simpler version of other variant.
    Counts trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  

def load_test_report_lossAcc_for_model(test_data, model, learning_rate=1e-3, bs=10000):
    """Loads models (all folds) def by train_id, name_prefix, modelName and L.
    Loads test set of right size."""
    
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(
        test_data, batch_size=bs, sampler=test_sampler)
    results = np.zeros(1, 2)

    criterion = nn.BCELoss(reduction='sum')
    num_samples = len(test_data)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model_loss, model_acc = 0, 0

    model.eval()  # eval mode, not model.train() for train mode
    with torch.no_grad():  # no grad calculations
        for data, label in test_loader:
            output = model(data)
            batch_loss = criterion(output, label)

            # threshold value of 0.5 to determine class.
            batch_acc = ((output > 0.5).float() == label).float().sum().item()
            model_acc += batch_acc / num_samples
            model_loss += batch_loss.item() / num_samples

    results[model_id] = model_loss, model_acc
    print(f"Succesfully tested {model.name} model with an average acc of ",
          f"{results[:,1].mean()*100:0.2f}%")
    return results