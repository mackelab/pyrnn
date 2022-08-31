import numpy as np
from itertools import permutations
import torch
from torch.utils.data import Dataset


class DMS(Dataset):
    def __init__(self, task_params):
        """
        Initialize a Delayed Match to Sample task
        with arbitrary sequence length

        Trials are structured with the following periods:
            Onset, Stimulus, Delay, Probe, Response

        Args:
            task_params: dictionary containing task parameters
        """

        if task_params["n_channels"] < 2:
            print("WARNING: use minimum of 2 input channels")
        if task_params["n_channels"] <= task_params["seq_len"]:
            print("Sequence length has to be smaller equal input channels")

        self.task_params = task_params

        # generate an array listing all posible trials
        # for this we simply take all permutations such that
        # index indicates stimulus position and value indicates stimulus identity
        # e.g. [2, 5, 0] is a trial with Stimulus 2 followed by Stimulus 5 followed by Stimulus 0

        self.all_trials_arr = np.array(
            list(
                set(
                    permutations(
                        np.arange(task_params["n_channels"]),
                        int(task_params["seq_len"]),
                    )
                )
            )
        )
        self.n_trials = self.all_trials_arr.shape[0]

    def __len__(self):
        """Returns number of possible stim combinations (match + no match)"""
        return self.n_trials * 2

    def __getitem__(self, idx):
        """
        Returns 1 trial with given idx

        Args:
            idx, trial index

        Returns:
            trial, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """

        trial, label, rand_delay, rand_ons = self.generate_stim(idx)
        target, mask = self.generate_target(label, rand_delay, rand_ons)
        return trial, target, mask

    def trial_len(self):
        """total trial duration"""
        trial_len = (
            self.probe_end()
            + self.task_params["response_ons"]
            + self.task_params["response_dur"]
        )
        return trial_len

    def probe_end(self):
        """time at which probe is done"""
        probe_duration = self.task_params["seq_len"] * self.task_params["probe_dur"] + (
            self.task_params["seq_len"] - 1
        ) * (self.task_params["probe_offs"])
        return self.probe_start() + probe_duration

    def probe_start(self):
        """time at which probe starts"""
        return self.delay_start() + self.task_params["delay"]

    def delay_start(self):
        """time at which delay starts"""
        stim_duration = self.task_params["seq_len"] * (self.task_params["stim_dur"]) + (
            self.task_params["seq_len"] - 1
        ) * (self.task_params["stim_offs"])
        return stim_duration + self.task_params["stim_ons"]

    def generate_stim(self, idx):
        """
        Generates one trial

        Args:
            idx: trial index
        Returns:
            u, trial Tensor of size [seq_len, n_inp]
            label, either 1 (match) or 0 (non-match)
            rand_delay, how much delay was shifted
            rand_ons, how much onset was shifted
        """

        # first half of trials are match, second half non match
        if idx >= len(self.all_trials_arr):
            idx -= len(self.all_trials_arr)
            label = 0
        else:
            label = 1
        stim = self.all_trials_arr[idx]

        T = self.trial_len()

        # u is the trial array
        u = torch.zeros((T, self.task_params["n_channels"]), dtype=torch.float32)

        # randomize delay period
        if self.task_params["rand_delay"]:
            rand_delay = torch.randint(self.task_params["rand_delay"], size=(1,))

        else:
            rand_delay = 0

        # randomize stimulus onset
        if self.task_params["rand_ons"]:
            rand_ons = torch.randint(self.task_params["rand_ons"], size=(1,))
        else:
            rand_ons = 0

        # generate trial
        stim_on = self.task_params["stim_ons"] - rand_ons
        probe_on = self.probe_start() - rand_ons - rand_delay

        # if match trial: probe equals stimulus, otherwise
        # probe is shuffled
        if label:
            probe = stim
        else:
            probe = self.non_match_probe(stim)

        # set stims to 1
        for i in range(self.task_params["seq_len"]):
            stim_start = (
                stim_on
                + (self.task_params["stim_dur"] + self.task_params["stim_offs"]) * i
            )
            probe_start = (
                probe_on
                + (self.task_params["probe_dur"] + self.task_params["probe_offs"]) * i
            )
            u[stim_start : stim_start + self.task_params["stim_dur"], stim[i]] = 1
            u[probe_start : probe_start + self.task_params["probe_dur"], probe[i]] = 1

        return u, label, rand_delay, rand_ons

    def generate_target(self, label, rand_delay, rand_onset):
        """
        Generate target and mask matching input

        Args:
            label, either 1 (match) or 0 (non-match)
            rand_delay, how much delay was shifted
            rand_ons, how much onset was shifted

        Returns:
            z, target Tensor of size [seq_len, n_out]
            mask, Tensor of size [seq_len, n_out]

        """
        T = self.trial_len()

        # output array
        z = torch.zeros((T, 1), dtype=torch.float32)

        # mask array, to put emphasis on specific time points
        mask = torch.zeros((T, 1), dtype=torch.float32)

        response_time = (
            self.probe_end()
            + self.task_params["response_ons"]
            - rand_delay
            - rand_onset
        )
        if label:
            target = 1
        else:
            target = -1
        z[response_time : response_time + self.task_params["response_dur"]] = target

        mask[response_time : response_time + self.task_params["response_dur"]] = 1

        return z, mask

    def non_match_probe(self, stim):
        """shuffle probe till not matching stimulus"""
        if self.task_params["seq_len"] > 1:
            probe = np.copy(stim)
            while np.array_equal(probe, stim):
                # use torch random function!
                idx = torch.randperm(len(probe)).numpy()
                probe = probe[idx]
        else:
            probe = (np.copy(stim) - 1) * -1
        return probe
