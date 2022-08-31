def default_params():
    
    """Returns dictionary with params used in train_rnn.ipynb"""
    
    n_stims = 8  # number of potential stimuli

    model_params = {
        "nonlinearity": "tanh",           # activation function
        "rank": 0,                        # rank, set to 0 for full rank
        "n_inp": n_stims,                 # amount of input units
        "p_inp": 1,                       # probability of connection input
        "n_rec": 200,                     # amount of recurrent units
        "p_rec": 1,                       # probability of connection recurrent
        "n_out": 1,                       # amount of output units
        "scale_w_inp": 1,                 # scale input weights
        "scale_w_out": 1,                 # scale output weights
        "w_rec_dist": "gauss",            # recurrent weight dist, gauss or gamma
        "spectr_rad": 1.5,                # gain param, recurrent weights
        "spectr_norm": False,              # use spectral normalisation on rec weights
        "apply_dale": False,               # only exitatory / inhibitory outgoing conns
        "p_inh": 0.2,                     # probability of inhibitory connection
        "balance_dale": True,             # expected input to units (per model) is 0
        "row_balance_dale": False,        # expected input to units (per unit) is 0
        "1overN_out_scaling": False,      # deep versus shallow learning? (not well tested)
        "train_w_inp": False,              # train input weights
        "train_w_inp_scale": False,       # train input scaling factor
        "train_w_rec": True,              # train recurrent weights
        "train_taus": False,              # train time constants
        "train_w_out": True,              # train output weights
        "train_w_out_scale": False,       # train output scaling factor
        "train_x0": False,                 # train initial state
        "tau_lims": [100],                # tau limits (min, max) or (value) in ms
        "dt": 10,                         # timestep in ms
        "noise_std": 0.05,                # noise std
        "scale_x0": 0.1,                  # std of initial state, if gaussian
    }

    training_params = {
        "n_epochs": 50,                  # number of passes through possible trials
        "lr": 1e-4,                       # learning rate
        "batch_size": 128,                  # batch size
        "clip_gradient": 1,               # to avoid explosion of gradients
        "cuda": True,                     # train on GPU if True
        "loss_fn": "mse",                 # loss function (mse, cos or none)
        "optimizer": "adam",              # optimizer (adam)
        "osc_reg_cost": 0,                # oscillatory regularisation weight
        "osc_reg_freq": 2,                # oscillatory regularisation frequency
        "offset_reg_cost": 0,             # offset regularisation cost

    }

    task_params = {
        "stim_ons": 40,                   # stimulus onset (units are number of time steps)
        "rand_ons": 0,                    # randomise onset with this amount
        "stim_dur": 20,                   # stimulus duration
        "stim_offs": 20,                  # stimulus offsets
        "delay": 25,                     # delay length
        "rand_delay": 0,                  # randomize delay with this amount
        "probe_dur": 20,                  # probe duration
        "probe_offs": 20,                 # probe offset
        "response_dur": 40,               # response duration
        "response_ons": 20,               # response onset
        "seq_len": 4,                     # sequence length
        "n_channels": n_stims,            # number of stimulus
    }
    return model_params, training_params, task_params