import torch
import torch.nn as nn
import numpy as np
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from initializers import *

class RNN(nn.Module):
    def __init__(self, params):
        """
        Initializes a biologically inspired recurrent neural network model

        Args:
            params: python dictionary containing network parameters
        """

        super(RNN, self).__init__()
        self.params = params

        # choose between full and low rank RNN cell
        
        if params["rank"]:
            self.rnn = LR_RNNCell(params)
        else:
            self.rnn = RNNCell(params)  # , bias = False)

        # hidden state at t = 0
        self.x0 = nn.Parameter(torch.Tensor(1, params["n_rec"]))
        if not params["train_x0"]:
            self.x0.requires_grad = False
        with torch.no_grad():
            if params["scale_x0"]:
                self.x0 = self.x0.normal_(std=params["scale_x0"])
            else:
                self.x0 = self.x0.copy_(
                    torch.zeros(1, params["n_rec"], dtype=torch.float32)
                )

    def forward(self, input, x0=None):
        """
        Do a forward pass through all time steps

        Args:
            input: tensor of size [batch_size, seq_len, n_inp]
            x0 (optional): hidden state at t=0
        """

        batch_size = input.size(0)
        seq_len = input.size(1)

        # precompute noise and allocate output tensors
        noise = (
            torch.randn(
                batch_size,
                seq_len,
                self.params["n_rec"],
                device=self.rnn.w_inp.device,
                dtype=torch.float32,
            )
            * self.params["noise_std"]
        )
        outputs = torch.zeros(
            batch_size,
            seq_len,
            self.params["n_out"],
            device=self.rnn.w_inp.device,
            dtype=torch.float32,
        )
        rates = torch.zeros(
            batch_size,
            seq_len,
            self.params["n_rec"],
            device=self.rnn.w_inp.device,
            dtype=torch.float32,
        )

        # initialize current at t=0
        if x0 is None:
            h_t = torch.tile(self.x0, dims=(batch_size, 1))
        else:
            if x0.shape[0] == 1:
                h_t = torch.tile(x0, dims=(batch_size, 1))
            else:
                h_t = x0

        # run through all timesteps
        for i, input_t in enumerate(input.split(1, dim=1)):
            h_t, output = self.rnn(input_t.squeeze(dim=1), h_t, noise[:, i])
            rates[:, i] = h_t
            outputs[:, i] = output

        return rates, outputs


class RNNCell(nn.Module):
    def __init__(self, params):
        """
        Full rank RNN cell (contains parameters and computes
        one step forward)

        args:
            params: dictionary with model params
        """
        super(RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params)

        # declare network parameters
        self.w_inp = nn.Parameter(torch.Tensor(params["n_inp"], params["n_rec"]))
        self.w_rec = nn.Parameter(torch.Tensor(params["n_rec"], params["n_rec"]))
        self.w_out = nn.Parameter(torch.Tensor(params["n_rec"], params["n_out"]))
        
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))   
        self.w_out_scale = nn.Parameter(torch.Tensor(1))
        
        if not params["train_w_inp"]:
            self.w_inp.requires_grad = False
        if not params["train_w_rec"]:
            self.w_rec.requires_grad = False
        if not params["train_w_out"]:
            self.w_out.requires_grad = False
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False
        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False

        # time constants
        self.dt = params["dt"]
        self.tau = params["tau_lims"]
        if len(params["tau_lims"]) > 1:
            self.taus_gaus = nn.Parameter(torch.Tensor(params["n_rec"]))
            if not params["train_taus"]:
                self.taus_gaus.requires_grad = False

        # initialize parameters
        with torch.no_grad():
            
            w_inp = initialize_w_inp(params)
            self.w_inp = self.w_inp.copy_(torch.from_numpy(w_inp))
           
            w_rec, dale_mask = initialize_w_rec(params)
            self.dale_mask = torch.from_numpy(dale_mask)
            self.w_rec = self.w_rec.copy_(torch.from_numpy(w_rec))
           
            # deep versus shallow learning?
            if params["1overN_out_scaling"]:
                self.w_out = self.w_out.normal_(
                    std=params["scale_w_out"] / params["n_rec"]
                )
            else:
                self.w_out = self.w_out.normal_(
                    std=params["scale_w_out"] / np.sqrt(params["n_rec"])
                )
          

            # connection mask
            if params["apply_dale"]:
                self.mask = mask_dale
            else:
                self.mask = mask_none

            # possibly initialize tau with distribution
            # (this is then later projected to be within preset limits)
            if len(params["tau_lims"]) > 1:
                self.taus_gaus.normal_(std=1)
           
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])
           


    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, seq_len, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, n_rec]
            noise: noise at current time step, tensor of size [batch_size, n_rec]

        Returns:
            x: hidden state at next time step, tensor of size [batch_size, n_rec]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        # apply mask to weight matrix
        w_eff = self.mask(self.w_rec, self.dale_mask)

        # compute alpha (dt/tau), and scale noise accordingly
        if len(self.tau) == 1:
            alpha = self.dt / self.tau[0]
            noise_t = np.sqrt(2 * alpha) * noise
        else:
            taus_sig = project_taus(self.taus_gaus, self.tau[0], self.tau[1])
            alpha = self.dt / taus_sig
            noise_t = torch.sqrt(2 * alpha) * noise

        # calculate input to units
        rec_input = torch.matmul(self.nonlinearity(x), w_eff.t()) + input.matmul(
            self.w_inp*self.w_inp_scale
        )
        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + noise_t

        # linear readout of the rates
        output = self.nonlinearity(x).matmul(self.w_out*self.w_out_scale)

        return x, output


class LR_RNNCell(nn.Module):
    def __init__(self, params):

        """
        RNN cell with rank of the recurrent weight matrix constrained
        (contains parameters and computes one step forward)

        args:
            params: dictionary with model params
        """
        super(LR_RNNCell, self).__init__()

        # activation function
        self.nonlinearity = set_nonlinearity(params)

        # declare network parameters
        self.w_inp = nn.Parameter(torch.Tensor(params["n_inp"], params["n_rec"]))
        if not params["train_w_inp"]:
            self.w_inp.requires_grad = False
        self.w_inp_scale = nn.Parameter(torch.Tensor(1))
        if not params["train_w_inp_scale"]:
            self.w_inp_scale.requires_grad = False

        self.m = nn.Parameter(torch.Tensor(params["n_rec"], params["rank"]))
        if not params["train_m"]:
            self.m.requires_grad = False
        self.n = nn.Parameter(torch.Tensor(params["rank"], params["n_rec"]))
        if not params["train_n"]:
            self.n.requires_grad = False

        self.w_out = nn.Parameter(torch.Tensor(params["n_rec"], params["n_out"]))
        if not params["train_w_out"]:
            self.w_out.requires_grad = False
        self.w_out_scale = nn.Parameter(torch.Tensor(1, params["n_out"]))
        if not params["train_w_out_scale"]:
            self.w_out_scale.requires_grad = False

        self.dt = params["dt"]
        self.tau = params["tau_lims"][0]
        if len(params["tau_lims"]) > 1:
            print("WARNING: distribution of Tau currently not supported for LR RNN")
        self.N = params["n_rec"]

        # initialize network parameters
        with torch.no_grad():

            if params["loadings"] is None:
                loadings = initialize_loadings(params)
            else:
                loadings = params["loadings"]

            self.w_inp = self.w_inp.copy_(torch.from_numpy(loadings[: params["n_inp"]]))
            self.w_out = self.w_out.copy_(
                torch.from_numpy(loadings[-params["n_out"] :]).T
            )

            # n and m are the left and right singular vectors
            # of the recurrent weight matrix

            self.n = self.n.copy_(
                torch.from_numpy(
                    loadings[params["n_inp"] : params["n_inp"] + params["rank"]]
                    * np.sqrt(params["scale_n"])
                )
            )
            self.m = self.m.copy_(
                torch.from_numpy(
                    loadings[
                        params["n_inp"]
                        + params["rank"] : params["n_inp"]
                        + params["rank"] * 2
                    ].T
                    * np.sqrt(params["scale_m"])
                )
            )
            self.w_inp_scale = self.w_inp_scale.fill_(params["scale_w_inp"])
            self.w_out_scale = self.w_out_scale.fill_(params["scale_w_out"])

    def forward(self, input, x, noise=0):
        """
        Do a forward pass through one timestep

        Args:
            input: tensor of size [batch_size, n_inp]
            x: hidden state at current time step, tensor of size [batch_size, n_rec]
            noise: noise at current time step, tensor of size [batch_size, n_rec]

        Returns:
            x: hidden state at next time step, tensor of size [batch_size, n_rec]
            output: linear readout at next time step, tensor of size [batch_size, n_out]

        """

        alpha = self.dt / self.tau

        # input to units
        rec_input = torch.matmul(
            torch.matmul(self.nonlinearity(x), self.n.t()), self.m.t()
        ) / self.N + self.w_inp_scale * input.matmul(self.w_inp)
        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + np.sqrt(2 * alpha) * noise

        # linear readout
        output = self.w_out_scale * self.nonlinearity(x).matmul(self.w_out) / self.N

        return x, output




def predict(
    rnn,
    _input,
    loss_fn=None,
    _target=None,
    _mask=None,
    x0=None,
):
    """
    Do a forward pass with an RNN
    Utility function to call outside of training

    Args:
        rnn: Initialized RNN
        _input: input tensor of size [batch_size, seq_len, n_inp]
        loss_fn (optional): loss function
        _target (optional), tensor of size [batch_size, seq_len, n_out]
        _mask(optional), tensor of size [batch_size, seq_len, n_out]
        _x0(optional), tensor of size [batch_size, n_rec]

    Returns:
        rates: tensor of size [batch_size, seq_len, n_rec]
        predict: tensor of size [batch_size, seq_len, n_out]

    """
    rnn.eval()
    # if single trial, add batch dimension
    if _input.dim() < 3:
        _input = _input.unsqueeze(0)

    device = rnn.rnn.w_inp.device
    input = _input.to(device=device)

    if loss_fn is not None:
        if _target.dim() < 3:
            _target = _target.unsqueeze(0)
        if _mask.dim() < 3:
            _mask = _mask.unsqueeze(0)
        target = _target.to(device=device)
        mask = _mask.to(device=device)

    with torch.no_grad():
        rates, predict = rnn(input, x0=x0)

        if loss_fn is not None:
            loss = loss_fn(predict, target, mask)
            print("test loss:", loss.item())
            print("==========================")

    return rates.cpu().detach().numpy(), predict.cpu().detach().numpy()


def project_taus(x, lim_low, lim_high):
    """
    Apply a non linear projection map to keep x within bounds

    Args:
        x: Tensor with unconstrained range
        lim_low: lower bound on range
        lim_high: upper bound on range

    Returns:
        x_lim: Tensor constrained to have range (lim_low, lim_high)
    """

    x_lim = torch.sigmoid(x) * (lim_high - lim_low) + lim_low
    return x_lim


# Note make these callable classes so we don't have to pass the mask each time?
# implement __cal__ method
def mask_dale(w_rec, mask):
    """Apply Dale mask"""
    return torch.matmul(torch.relu(w_rec), mask)


def mask_none(w_rec, mask):
    """Apply no mask"""
    return w_rec


def set_nonlinearity(params):
    """utility returning activation function"""
    if params["nonlinearity"] == "tanh":
        return torch.tanh
    elif params["nonlinearity"] == "identity":
        return lambda x: x
    elif params["nonlinearity"] == "relu":
        return nn.ReLU()
    elif params["nonlinearity"] == "softplus":
        softplus_scale = 1  # Note that scale 1 is quite far from relu
        nonlinearity = (
            lambda x: torch.log(1.0 + torch.exp(softplus_scale * x)) / softplus_scale
        )
        return nonlinearity
    elif type(params["nonlinearity"]) == str:
        print("Nonlinearity not yet implemented.")
        print("Continuing with identity")
        return lambda x: x
    else:
        return params["nonlinearity"]
