import os
from glob import glob
from random import shuffle
import numpy as np
import re
import torch
from torch.cuda import device_count
from torch.utils.data import Dataset
from NNClasses import sawtooth, sawtooth_normalized
from torch.nn.functional import conv2d
# __getitem__ returns T as well

class CustomDataset(Dataset):
    """Custom Data set for use with pytorch Neural networks.

    Args:
        root(string): Location of data, additional data can be added later by
            calling load_configs
        L(int): Size of the systems
        Tc0(float): initial value of critical temperature
        data_size(int)=None: limits size of data. Defaults to None with no limit.
        type_(string)="multiClass": "multiClass" for labels [1,0] or [0,1], labels.size = N,1 
                    or "binary" for labels [0] or [1], labels.size = N
        load_in_order(Bool)=False: shuffles data after loading configuration if False
        dataFromJulia(Bool)=True: determines how configs are loaded
    """
    # With features: T, L, config

    def __init__(self, root, L, Tc0, data_Size=None, type_="multiClass", normalize=False,
                 load_in_order=False, dataFromJulia=True, transform=None, target_transform=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.type_ = type_
        self.Tc = Tc0
        self.L = L
        self.max_size = data_Size  # testing purposes
        self.state = "spin angles"
        self.root = root
        self.normalized_data = normalize  # if data has to be normalized after loading
        self.Ts = torch.empty(0, 1).to(self.device)
        self.configs = torch.empty(0, L, L).to(self.device)
        self.labels = torch.empty(0, 1).to(self.device)

        self.transform = transform
        self.target_transform = target_transform
        if dataFromJulia:
            # julia folder order is data/new/train-1000/L=L/configs.npy
            _root = os.path.join(root, f"L={L}", "configs.npy")
            self.load_configsJulia(_root, load_in_order)
        else:
            self.load_configs(root, load_in_order)

        if self.normalized_data:
            self.configs /= (2*np.pi)

    @classmethod
    def usingData(cls, L, configs, Ts, Tc=0.894, normalize=False, state="spin angles", device=None):
        obj = cls.__new__(cls)  # Does not call __init__
        # Don't forget to call any polymorphic base class initializers
        super(CustomDataset, obj).__init__()
        if device is None:
            obj.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            obj.device = device
            
        obj.configs = torch.from_numpy(configs).float().to(obj.device)
        obj.Ts = torch.from_numpy(Ts).float().to(obj.device)
        obj.Tc = Tc
        obj.L = L
        obj.state = state
        obj.type_ = "multiClass"
        obj.root = ""
        obj.recalc_labels()
        obj.normalized_data = normalize
        obj.max_size = len(obj)
        return obj

    def __len__(self):
        return self.configs.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        x = self.configs[idx, :, :]
        if self.type_ == "binary":
            # binary -> labels.size = [N]
            y = self.labels[idx]
        else:  # if self.type_ == "multiClass":
            # labels.size = [N,1]
            y = self.labels[idx, :]
        return x, y, self.Ts[idx]

    def load_configs(self, path, load_in_order):
        """Loads configs at location root+path.
        path: location to folder where configs are saved."""

        def pattern(s): return r'^.*' + s + r'=([0-9.]+).*$'
        # [ff.path for ff in os.scandir(path) if os.path.isfile(ff)]
        fileList = glob(os.path.join(path, "**", "*.txt"), recursive=True)
        if not load_in_order:
            shuffle(fileList)

        for f in fileList:
            L = int(re.sub(pattern('L'), r'\1', f))

            if (L == self.L):
                T = float(re.sub(pattern('T'), r'\1', f))

                # config, label and T add
                # GOOGLE COLLAB: PATH IS DIFFERENT!!!!
                config = torch.from_numpy(
                    np.loadtxt(f)).float().to(self.device).unsqueeze(0)
                label = torch.FloatTensor([[1.]] if T >= self.Tc else [
                                          [0.]]).to(self.device)
                Ttens = torch.FloatTensor([[T]]).to(self.device)

                self.configs = torch.cat((self.configs, config), 0)
                self.labels = torch.cat((self.labels, label))
                self.Ts = torch.cat((self.Ts, Ttens))

                if self.max_size:
                    if len(self) >= self.max_size:
                        break

    def load_configsJulia(self, path, load_in_order):
        """Loads configs at location root+path.
        path: dir_name in generate-xy.jl
        load is always in order.
        """
        self.configs = np.load(path)
        self.configs = torch.from_numpy(self.configs.reshape(
            self.configs.shape[0], self.L, self.L)).to(self.device).type(torch.FloatTensor)
        self.Ts = torch.from_numpy(np.repeat(np.linspace(0.1, 3.0, 64),
                                             self.configs.size()[0]//64)).to(self.device).type(torch.FloatTensor)

        self.labels = self.Ts >= self.Tc
        self.labels = torch.FloatTensor([[1.] if isBiggerBool else [0.]
                                         for isBiggerBool in self.labels]).to(self.device)

        if load_in_order:
            print(
                "Loading configs in order, by default, this means sorted by temperature.")
        else:
            print("Data shuffled.")
            self.shuffleData()

        if self.max_size is None or len(self) <= self.max_size:
            self.max_size = len(self)
        else:
            # shorten data set
            self.labels = self.labels[:self.max_size]
            self.configs = self.configs[:self.max_size]
            self.Ts = self.Ts[:self.max_size]

    def to(self, device):
        self.device = device
        self.configs = self.configs.to(self.device)
        self.labels = self.labels.to(self.device)
        self.Ts = self.Ts.to(self.device)

    def shortenData(self, size):
        """Shortens data to size if dataset is bigger, updates max_size pm if data is shortened."""
        if self.max_size > size:
            self.labels = self.labels[:size]
            self.configs = self.configs[:size]
            self.Ts = self.Ts[:size]
            self.max_size = size
    
    def limit_size_per_T(self, max_per_T):
        """Limits dataset in size, to 64*max_per_T samples.
        max_per_T cannot exceed 1000."""
        output, inverse_indices = self.Ts.unique(return_inverse=True)

        assert max_per_T <= 1000, "Maximum 1000 samples per temperature."
        
        new_Ts = torch.empty(len(output)*max_per_T, device=self.device)
        new_cfgs = torch.empty(len(output)*max_per_T, self.L, self.L, device=self.device)
        for T_i, T in enumerate(output):
          bools = self.Ts == T
          temp = self.Ts[bools][:max_per_T]
          
          assert temp.shape[0] == max_per_T, f"Not found enough samples for T={T}."
          
          new_Ts[max_per_T*T_i:max_per_T*(T_i+1)] = temp
          new_cfgs[max_per_T*T_i:max_per_T*(T_i+1)] = self.configs[bools][:max_per_T]
        
        self.Ts = new_Ts
        self.configs = new_cfgs
        self.recalc_labels()
        self.max_size = len(self)
        print(f"Limited data set to {max_per_T*len(output)} samples, or {max_per_T} per T.")

    def shuffleData(self):
        """Shuffles data around.
        This includes configs, Ts and labels.
        """
        order = torch.randperm(len(self))
        self.configs = self.configs[order, :, :]
        self.labels = self.labels[order, :]
        self.Ts = self.Ts[order]

    def set_type(self, new_type):
        """Sets type of labels to multiClass or binary.

        Args:
            new_type (string): "binary" or "multiClass". 
             "multiClass" for labels [1,0] or [0,1], labels.size = N,1 
             or "binary" for labels [0] or [1], labels.size = N
        """
        if new_type != self.type_ and new_type in ["binary", "multiClass"]:
            self.type_ = new_type
            if self.type_ == "binary":
                self.labels = self.labels.squeeze()
            elif self.type_ == "multiClass":
                self.labels = self.labels.unsqueeze(1)

    def print_RandomDataPoint(self, n=1, i=0):
        """Prints n random data points (label and T), indices can be specified with array i"""
        if type(i) == int:
            i = np.random.randint(0, high=len(self), size=n)
        for idx in i:
            print(f"idx {idx}: {self.labels[idx]}\t{self.Ts[idx]}")

    def get_X(self):
        return self.configs

    def get_y(self):
        return self.labels

    def recalc_labels(self, Tc=None):
        if Tc is not None:
            self.Tc = Tc
        self.labels = torch.empty(self.Ts.shape, dtype=float).to(self.device)
        self.labels = torch.logical_not(self.Ts < self.Tc, out=self.labels)

    def normalize_data(self):
        """Normalizes data if not already done so.
        This cannot be done twice on the same dataset.
        """
        if not self.normalized_data:
            print("Data normalized.")
            self.configs /= (2*np.pi)
            self.normalized_data = True

    def get_vortex_from_spin(self):
        """Returns vortices from raw spin configurations.
        Transform done using pytorch tensors.
            1. apply 4 filters
            2. map the 4 angle differences using sawtooth
            3. sum up
        """
        # only if starting from spin angles:
        if self.state == "spin angles":
            print("vortex from spin device:", self.device)
            # 4 filters for the 4 neighbors
            K1 = torch.FloatTensor([[-1., 1.], [0., 0.]]).unsqueeze(0)
            K2 = torch.FloatTensor([[0., -1.], [0., 1.]]).unsqueeze(0)
            K3 = torch.FloatTensor([[1., 0.], [-1., 0.]]).unsqueeze(0)
            K4 = torch.FloatTensor([[0., 0.], [1., -1.]]).unsqueeze(0)
            Ks = torch.cat((K1, K2, K3, K4)).unsqueeze(1).to(self.device)

            # include pbc for configs of shape M, L, L
            temp = self.configs.repeat(1, 2, 2)[:, 0:self.L+1, 0:self.L+1]

            # apply filters
            temp = conv2d(temp.unsqueeze(1), weight=Ks)

            # apply sawtooth, remap spins
            if self.normalized_data:
                temp = sawtooth_normalized(temp)
            else:
                temp = sawtooth(temp)
                temp /= 2*np.pi

            # sum up
            temp = conv2d(temp, torch.ones(
                (1, 4, 1, 1)).float().to(self.device)).squeeze(0)

            return temp

        elif self.state == "vortices":
            # if already transformed.
            return self.configs

        else:
            # state unaccounted for
            raise ValueError(f"Invalid data state: {self.state}")

    def get_vortex_dens_from_configs(self):
        """Returns np array of vortex densities of all configs.
        """
        if self.state != "vortices":
            temp = self.get_vortex_from_spin()
        else:
            temp = self.configs

        temp = abs(temp) / (2*np.pi)

        vort = np.array([t.sum()/(2*self.L**2) for t in temp])
        return vort

    def spin_to_vortex(self):
        """Transforms raw spin configurations into vortices.
        Transform done using pytorch tensors.
        See 'get_vortex_from_spin' for the implementation.
        """
        # only if not already transformed
        if self.state == "spin angles":
            self.configs = self.get_vortex_from_spin()

            # remember the state of the configs
            self.state = "vortices"

    def print_length(self):
        print(f"Data set has size {len(self)}")

    def get_T_distribution(self):
        """Calculates how many samples are there for each
        present T in the dataset

        Returns:
            ([float], [int]): np array of Ts and how many of each
        """
        T_range = self.Ts.unique(sorted=True).numpy()

        num_samples = np.zeros(T_range.shape)

        for i, T in enumerate(T_range):
            # sum samples where T == T, and get item from 1 value tensor
            num_samples[i] = (self.Ts == T).sum().item()

        return T_range, num_samples

    def plot_T_distribution(self, ax):
        """Plots the T distribution of the dataset on the Axes.

        Args:
            ax (Axes): An axes object onto which the T distr
                       is plotted.
        """
        # get data
        T_range, num_samples = self.get_T_distribution()

        # using plot because ~64 T values is too much for hist
        ax.plot(T_range, num_samples)

        # decoration
        ax.set_xlabel("T")
        ax.set_ylabel("Number of samples")
