import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi
from copy import deepcopy
import warnings


class FFNN(nn.Module):
    def __init__(self, L, normalized_data_bool=False, device=None):
        super(FFNN, self).__init__()
        # 1 input spin lattice channel, 1 output channel
        # L = size of grid
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.L = L
        self.name = "FFNN"
        self.fc1 = nn.Linear(L*L, 1024, device=self.device)
        self.fc2 = nn.Linear(1024, 1, device=self.device)

    def longname(self):
        return self.name
        
    def forward(self, x):
        if x.dim() == 4:
            x.squeeze_(1)  # N,1,L,L -> N,L,L
        # reshape: N,L,L -> N,L²
        x = x.view(x.size(0), -1)  # x.size(0) to keep batch size
        # hidden layer with relu activation layer
        x = torch.sigmoid(self.fc1(x))
        # output via single sigmoid unit
        x = torch.sigmoid(self.fc2(x))
        # Because we use withlogitsloss, we don't need
        # the final sigmoid.
        return x


class CNN(nn.Module):
    """1 input spin lattice channel, 1 output channels
    conv1 (3x3, s=1, # = 8)
    input should be of shape
    input:  (N=batch_size,  C_in=1,   L,   L)
    output: (N=batch_size, C_out=8, L-2, L-2)"""

    def __init__(self, L, normalized_data_bool=False, device=None):
        super(CNN, self).__init__()
        self.L = L
        self.name = "CNN"
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        else:
          self.device = device

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, device=self.device)
        # conv2 (3x3, s=1, # = 16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, device=self.device)
        self.fc1 = nn.Linear(16*(L//2 - 2)**2, 32, device=self.device)
        self.fc2 = nn.Linear(32, 1, device=self.device)

    def longname(self):
        return self.name

    def forward(self, x):
        # reshape

        if x.dim() == 3:
            x.unsqueeze_(1)  # N,L,L -> N,1,L,L

        # conv layers
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        # max pool and reshape for fc layers
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # x.size(0) to keep batch size

        # fc layers
        x = F.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))
        return x


class Custom(CNN):
    def __init__(self, L, vortexInit=True, save_intermittent=False, normalized_data_bool=False, device=None):
        # The CNN model
        super(Custom, self).__init__(L, device=device)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # first conv layer
        # custom conv2d filters to calculate nieghboring spin angle differences
        # req_gradient to add to param list
        self.L = L
        self.name = "CustomNN"
        self.vortexInit = vortexInit

        # saving intermediate layer
        self.save_intermittent = save_intermittent
        self.intermediate_data = None

        # using normalized data or not
        self.normalized_data_bool = normalized_data_bool

        # Weights has shape 4,1,2,2, biases start out 0
        self.convVortex = nn.Conv2d(1,4, kernel_size=2, stride=1, device=self.device)
        self.reset_weights_convVortex()

    def reset_weights_convVortex(self):
        """Resets pms of convVortex layer to initial values
        based on self.vortexInit."""
        with torch.no_grad():
            self.convVortex.weight = nn.Parameter(self.initialWeights().to(self.device))
            self.convVortex.bias = nn.Parameter(self.initialBias().to(self.device))

    def initialBias(self):
        """Return tensor filled with zeros of shape 4.
        No device is given."""
        return torch.zeros((4)).float()

    def initialWeights(self):
        """Returns pytorch tensor for weights of convVortex,
        based on value of self.vortexInit.
        Weights shape 4,1,2,2.
        No device is given."""
        if self.vortexInit:
            K1 = torch.tensor(
                [[-1., 1.], [0., 0.]], requires_grad=True).float()
            K2 = torch.tensor(
                [[0., -1.], [0., 1.]], requires_grad=True).float()
            K3 = torch.tensor(
                [[1., 0], [-1., 0.]], requires_grad=True).float()
            K4 = torch.tensor(
                [[0., 0.], [1., -1.]], requires_grad=True).float()
        else: # randomInit
            K1 = (torch.rand(2, 2, dtype=float, requires_grad=True)*2-1).float()
            K2 = (torch.rand(2, 2, dtype=float, requires_grad=True)*2-1).float()
            K3 = (torch.rand(2, 2, dtype=float, requires_grad=True)*2-1).float()
            K4 = (torch.rand(2, 2, dtype=float, requires_grad=True)*2-1).float()
        
        return torch.cat(
            (K1.unsqueeze(0), K2.unsqueeze(0),
             K3.unsqueeze(0), K4.unsqueeze(0))).unsqueeze(1).float()


    def longname(self):
        return "CustomNN_vortexInit" if self.vortexInit else "CustomNN_randomInit"

    def forward(self, x):
        if x.dim() == 3:
          x.unsqueeze_(1) # N,L,L -> N,1,L,L

        if self.normalized_data_bool:
          assert torch.max(x).item() <= 1.1 and torch.min(x).item() >= -1.1  # ! assertion to test
        # padding periodic boundary in preparation for conv
        # N, 1, L, L -> N, 1, L+1, L+1
        x = x.repeat(1, 1, 2, 2)[:, :, 0:self.L+1, 0:self.L+1]

        # apply custom filters and use sawtooth fion to map to [-pi, +pi]
        x = self.convVortex(x) # N, 1, L+1, L+1 -> N, 4, L, L

        # remap using sawTooth function
        if self.normalized_data_bool:
            #minim, maxim = torch.min(x).item(), torch.max(x).item()
            #if minim < -1:
            #  print("minim", minim)
            #  print("weights", self.Weights)
            #if maxim > 1:
            #  print("maxim", maxim)
            #  print("weights", self.Weights)
            x = SawTooth_Normalized.apply(x)
        else:
            x = SawTooth.apply(x)

        # summation
        # N, 4, L, L -> N, L, L
        x = torch.sum(x, 1)

        # ! intercept x if needed
        if self.save_intermittent:
            self.intermediate_data = deepcopy(x.cpu().detach().numpy())

        # now pass through the CNN model
        x = super(Custom, self).forward(x)
        return x


def reset_weights(m, silence=False):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if type(layer) == nn.modules.conv.Conv2d and layer.weight.shape == torch.Size((4,1,2,2)): # custom layer convVortex
            if hasattr(m, "convVortex"):
                m.reset_weights_convVortex()
                if not silence:
                    print(f"reset trainable parameters of layer = {layer}")
                continue
            else:
                warnings.warn("attribute 'convVortex' expected but not found with model of type"+str(type(m)))

        if hasattr(layer, 'reset_parameters'):
            if not silence:
                print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


class SawTooth(torch.autograd.Function):
    """
    Applies the Sawtooth function as an activation function.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        return sawtooth(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        gradient of heaviside is always +1
        """
        return grad_output, None  # = grad_output * (+1)


class SawTooth_Normalized(torch.autograd.Function):
    """
    Applies the Sawtooth function as an activation function.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        return sawtooth_normalized(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        gradient of heaviside is always +1
        """
        return grad_output, None  # = grad_output * (+1)


def sawtooth(x):
    """Implements the saw function mapping x from
    [-2 pi, 2 pi] to [-pi, pi) (not including pi).

    Args:
        x (torch.tensor): Filled with x between -2 pi and 2 pi.

    Returns:
        x (torch.tensor): x but [-pi, pi).
    """
    device = x.device
    oneTensor = torch.FloatTensor([1.]).to(device)
    if (x.dtype == torch.double):
        oneTensor = oneTensor.double()

    H1 = torch.heaviside(x - pi, oneTensor).to(device)
    notH2 = torch.heaviside(x + pi, oneTensor).to(device)
    notH2 = - (notH2 - 1)
    out = x + 2 * pi * (notH2 - H1)
    return out.float()


def sawtooth_normalized(x):
    """sawtooth function normalized with factor 1/(2pi).
    Implements the saw function mapping x from
    [-1, 1] to [-0.5, 0.5) (not including 0.5).

    Args:
        x (torch.tensor): Filled with x between -1 and 1.

    Returns:
        x (torch.tensor): x but [-0.5, 0.5).
    """
    device = x.device
    oneTensor = torch.FloatTensor([1.]).to(device)
    if (x.dtype == torch.double):
        oneTensor = oneTensor.double()

    H1 = torch.heaviside(x - 0.5, oneTensor).to(device)
    notH2 = torch.heaviside(x + 0.5, oneTensor).to(device)
    notH2 = - (notH2 - 1)
    out = x + 2 * 0.5 * (notH2 - H1)
    return out.float()
