import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pickle
import time


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
            self.w_inp
        )
        # update hidden state
        x = (1 - alpha) * x + alpha * rec_input + noise_t

        # linear readout of the rates
        output = self.nonlinearity(x).matmul(self.w_out)

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



def train_rnn(rnn, training_params, task, sync_wandb=False, wandb_log_freq=100, x0=None):
    """
    Train a biologically inspired RNN

    Args:
        rnn: initialized RNN
        training_params: dictionary of training parameters
        task, Pytorch Dataset should on call return:
                            trial, of size [seq_len, n_inp]
                            target, of size [seq_len, n_out]
                            mask, of size [seq_len, n_out]
        syn_wandb (optional): Bool, indicates synchronsation with WandB
        wandb_log_freq: Int, how often to synchronise gradients + weights
    """

    dataloader = DataLoader(
        task, batch_size=training_params["batch_size"], shuffle=True
    )

    # cuda management, gpu highly speeds up training
    if training_params["cuda"]:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rnn.to(device=device)
    if rnn.params["apply_dale"]:
        rnn.rnn.dale_mask = rnn.rnn.dale_mask.to(device=device)

    # choose a loss function
    if training_params["loss_fn"] == "mse":
        loss_fn = mse_loss
    elif training_params["loss_fn"] == "cos":
        loss_fn = cos_loss
    elif training_params["loss_fn"] == "none":
        loss_fn = zero_loss
    else:
        print("WARNING: Loss function not implemented")
    
    reg_fns = []
    reg_costs = []
    # regulisation
    if training_params["osc_reg_cost"]:
        reg_fns.append(LFPLoss(
            freq=training_params["osc_reg_freq"],
            tstep=rnn.params["dt"] / 1000,
            T=dataloader.dataset[0][0].size(0),
            device=device,
        ))
        reg_costs.append(training_params["osc_reg_cost"])
    if training_params["offset_reg_cost"]:
        reg_fns.append(offset_loss)
        reg_costs.append(training_params["offset_reg_cost"])
    if len(reg_fns) ==0:
        reg_fns.append(zero_loss)
        reg_costs.append(0)
    reg_costs = torch.tensor(reg_costs, device = device)
    # optimizer
    if training_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(rnn.parameters(), lr=training_params["lr"])

    # initialize wandb
    if sync_wandb:
        wandb.init(
            project="phase-coding",
            group="pytorch",
            config={**rnn.params, **dataloader.dataset.task_params, **training_params},
        )
        config = wandb.config
        wandb.watch(rnn, log="all", log_freq=wandb_log_freq)

    # start timer before training
    time0 = time.time()
    # set rnn to training mode
    rnn.train()

    losses = []
    reg_losses = []
    # start training loop
    for i in range(training_params["n_epochs"]):
        loss_ep = 0.0
        reg_loss_ep = torch.zeros(len(reg_fns), device =device)
        num_len = 0

        # loop through dataloader
        for x, y, m in dataloader:

            x = x.to(device=device)
            y = y.to(device=device)
            m = m.to(device=device)

            rates, y_pred = rnn(x,x0)
            optimizer.zero_grad()
            task_loss = loss_fn(y_pred, y, m)
            reg_loss = torch.stack([reg_fn(rates, rnn.rnn) for reg_fn in reg_fns]).squeeze()#, device=device)
            #print(torch.sum(reg_loss*reg_costs))
            # grad descent
            loss = task_loss + torch.sum(reg_loss*reg_costs)
            loss.backward()

            # clip weights to avoid explosion
            if training_params["clip_gradient"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    rnn.parameters(), training_params["clip_gradient"]
                )

            # update weights
            optimizer.step()
            num_len += 1
            loss_ep += task_loss.item()
            reg_loss_ep += reg_loss#.tolist()

        # print average loss and print / sync
        loss_ep /= num_len
        reg_loss_ep /= num_len
        reg_loss_ep = reg_loss_ep.tolist()
        print(
            "epoch {:d} / {:d}: time={:.1f} s, task loss={:.5f}, reg loss=".format(
                i + 1,
                training_params["n_epochs"],
                time.time() - time0,
                loss_ep
                ) +
            str(["{:.5f}"]*len(reg_loss_ep)).format(*reg_loss_ep).strip("[]").replace("'", ""),          
            end="\r",
        )
        if sync_wandb:
            wandb.log({"task_loss": loss_ep, "reg_los": reg_loss_ep})
        losses.append(loss_ep)
        reg_losses.append(reg_loss_ep)
    print("\nDone. Training took %.1f sec." % (time.time() - time0))
    if sync_wandb:
        wandb.finish()
    rnn.eval()
    return losses, reg_losses


def load_rnn(name):
    """
    loads an RNN

    Args:
        name: String, path / name to where RNN is saved

    Returns:
        model: Initialized RNN
        params: dictionary of model parameters
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
    """

    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)
    with open(task_params_file, "rb") as f:
        task_params = pickle.load(f)
    with open(training_params_file, "rb") as f:
        training_params = pickle.load(f)

    model = RNN(params)
    model.load_state_dict(torch.load(state_dict_file))

    return model, params, task_params, training_params


def save_rnn(name, model, params, task_params, training_params):
    """
    saves an RNN

    Args:
        name: String, path / name to where RNN is saved
        model: Initialized RNN
        params: dictionary of model parameters
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
    """

    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"
    with open(params_file, "wb") as f:
        pickle.dump(params, f)
    with open(training_params_file, "wb") as f:
        pickle.dump(training_params, f)
    with open(task_params_file, "wb") as f:
        pickle.dump(task_params, f)

    torch.save(model.state_dict(), state_dict_file)


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


def initialize_w_rec(params):
    """
    Initializes (full rank) recurrent weight matrix

    Args:
        params: python dictionary containing network parameters

    Returns:
        w_rec: recurrent weight matrix, numpy array of shape [n_rec, n_rec]
        dale_mask: diagonal matrix indicating exh or inh, shape [n_rec, n_rec]
    """

    w_rec = np.zeros((params["n_rec"], params["n_rec"]), dtype=np.float32)
    dale_mask = np.eye(params["n_rec"], dtype=np.float32)
    rec_idx = np.where(
        np.random.rand(params["n_rec"], params["n_rec"]) < params["p_rec"]
    )

    # initialize with weights drawn from either Gaussian or Gamma distribution
    if params["w_rec_dist"] == "gauss":
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )
    elif params["w_dist"] == "gamma":
        w_rec[rec_idx[0], rec_idx[1]] = np.random.gamma(2, 0.5, len(rec_idx[0]))
        if params["spectr_norm"] == False:
            print(
                "WARNING: analytic normalisation not implemented for gamma, setting spectral normalisation to TRUE"
            )
            params["spectr_norm"] = True
        if params["apply_dale"] == False:
            print(
                "WARNING: Gamma distribution is all positive, use only with Dale's law, setting Dale's law to TRUE"
            )
            params["apply_dale"] == True

    else:
        print("WARNING: initialization not implemented, use Gauss or Gamma")
        print("continuing with Gauss")
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )

    # apply Dale's law, a neuron has either only exitatory
    # or only inhibitory outgoing connections
    if params["apply_dale"]:
        n_inh = int(params["n_rec"] * params["p_inh"])

        dale_mask[-n_inh:] *= -1
        w_rec = np.abs(w_rec)

        # Balanced DL (expectation input = 0)
        if params["balance_dale"]:
            EIratio = (1 - params["p_inh"]) / (params["p_inh"])
            w_rec[:, -n_inh:] *= EIratio
            # Row balanced DL (expectation input, per neuron, = 0)

            if params["row_balance_dale"]:
                ex_u = np.sum(w_rec[:, :-n_inh], axis=1)
                in_u = np.sum(w_rec[:, -n_inh:], axis=1)
                ratio = ex_u / in_u
                w_rec[:, :-n_inh] /= np.expand_dims(ratio, 1)
            b = np.sqrt((1 / (1 - (2 * params["p_rec"]) / np.pi)) / EIratio)
            w_rec *= b

    # set to desired spectral radius
    if params["spectr_norm"]:
        w_rec = (
            params["spectr_rad"]
            * w_rec
            / np.max(np.abs((np.linalg.eigvals(dale_mask.dot(w_rec)))))
        )
    print("spectral_rad: " + str(np.max(abs(np.linalg.eigvals(dale_mask.dot(w_rec))))))
    return w_rec, dale_mask


def initialize_w_inp(params):
    """
    Initializes input weight matrix

    Args:
        params: python dictionary containing network parameters

    Return:
        w_inp: input weight matrix, numpy array of size [n_rec, n_inp]
    """

    w_task = np.zeros((params["n_rec"], params["n_inp"]), dtype=np.float32)
    idx = np.array(
        np.where(np.random.rand(params["n_rec"], params["n_inp"]) < params["p_inp"])
    )
    w_task[idx[0], idx[1]] = np.random.randn(len(idx[0])) * np.sqrt(
        params["scale_w_inp"] / params["p_inp"]
    )

    return w_task.T


def initialize_loadings(params):
    """
    Initializes weight matrices for low rank networks

    Args:
        params: python dictionary containing network parameters

    Returns:
        loadings: weight matrices for low rank networks
            numpy array of size [rank * 2 + n_inp + n_out, n_rec]
    """

    n_loading = params["rank"] * 2 + params["n_inp"] + params["n_out"]

    if params["cov"] is None:
        # generate random covariance matrix
        # with n and m correlated to avoid vanishing gradients"
        cov = np.eye(n_loading)
        for i in range(params["rank"]):
            cov[params["n_inp"] + i, params["n_inp"] + params["rank"] + i] = 0.8
            cov[params["n_inp"] + params["rank"] + i, params["n_inp"] + i] = 0.8
    else:
        cov = params["cov"]
    # use cholesky decomposition to draw vectors
    chol_cov = np.float32(np.linalg.cholesky(cov))
    loadings = chol_cov @ np.float32(np.random.randn(n_loading, params["n_rec"]))

    return loadings


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


def extract_lfp(x, rnn_cell, normalize=True):
    """
    Calculate LFP as mean absolute synaptic input

    Args:
        x: currents throughout trials, Tensor of size [batch_size, seq_len, n_rec]
        rnn_cell: calculates forward pass of an RNN
        normalize(optional): zscore LFP

    Returns:
       lfp: local field potential, Tensor of size [batch_size, seq_len]
    """

    w_eff = rnn_cell.mask(rnn_cell.w_rec, rnn_cell.dale_mask)
    if len(rnn_cell.tau) > 1:
        tau = project_taus(rnn_cell.taus_gaus, rnn_cell.tau[0], rnn_cell.tau[1])
        alpha = rnn_cell.dt / tau
    else:
        alpha = rnn_cell.dt / rnn_cell.tau[0]

    # mean absolute synaptic input
    abs_inp = alpha * torch.matmul(rnn_cell.nonlinearity(x), torch.abs(w_eff.t()))
    lfp = torch.mean(abs_inp, dim=-1)

    if normalize:
        mean = torch.mean(lfp, dim=1).unsqueeze(1)
        var = torch.mean((lfp - mean.detach()) ** 2, dim=1).unsqueeze(1)
        lfp = (lfp - mean) / torch.sqrt(2 * var)

    return lfp

def offset_loss(rates, *args):
    """l2 reg on non zero mean single unit firing rates"""
    return torch.mean(torch.mean(rates[:,320:],dim = 1)**2)

class LFPLoss(object):
    def __init__(self, freq, tstep, T, device):
        """
        Regularizer to promote oscillations at specified frequency

        Args:
            freq: target freq in Hz
            tstep: timestep in S
            T: trial length in model steps
            device: cpu / cuda

        """
        trtime = np.arange(0, tstep * T, tstep, dtype=np.float32)[:T]
        sinF = torch.from_numpy(np.sin(freq * 2 * np.pi * trtime))
        cosF = torch.from_numpy(np.cos(freq * 2 * np.pi * trtime))
        self.sinF = sinF.to(device=device)
        self.cosF = cosF.to(device=device)
        self.T = T

    def __call__(self, x, rnn_cell):
        """
        Calculate loss as norm of fourier component

        Args:
            x: currents, Tensor of size [batch_size, seq_len, n_rec]
            rnn_cell: to calculate a forward pass
        """

        lfp = extract_lfp(x, rnn_cell)
        a = torch.tensordot(self.sinF, lfp, dims=[[0], [1]]) / self.T
        b = torch.tensordot(self.cosF, lfp, dims=[[0], [1]]) / self.T
        norm = torch.sqrt(a ** 2 + b ** 2)
        lfp_loss = 0.5 - torch.mean(norm)
        return lfp_loss


def zero_loss(x, *args):
    """
    Utility function returning zero
    Args:
        x: some tensor with correct device

    Returns:
        0

    """
    return torch.zeros(1, device=x.device)


def mse_loss(output, target, mask):
    """
    Mean squared error loss

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]
        mask, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    return loss


def cos_loss(output, target, mask):
    """
    Loss based on vector angle (needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]
        mask, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    criterion = nn.CosineSimilarity(dim=2)
    loss = 0.5 - 0.5 * ((mask.squeeze() * criterion(output, target)).sum() / mask.sum())
    return loss


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
