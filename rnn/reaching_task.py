import numpy as np
from itertools import permutations
import torch
from torch.utils.data import Dataset


class Reaching(Dataset):
    def __init__(self, task_params):
        """
        Initialize a Reaching task
        Args:
            task_params: dictionary containing task parameters
        """

        self.task_params = task_params
        


    def __len__(self):
        """Arbitrary number of trials, as they are randomly generated anyway"""
        return self.task_params['n_stim']

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """
        phase = np.pi*2*idx/self.__len__()
        inputs = torch.zeros(self.task_params['trial_len'],3)
        cp = np.cos(phase)
        sp = np.sin(phase)
        onset = torch.randint(low=self.task_params['onset'][0],high=self.task_params['onset'][1],size=(1,))[0]
        stim_dur = torch.randint(low=self.task_params['stim_dur'][0],high=self.task_params['stim_dur'][1],size=(1,))[0]
        delay_dur = torch.randint(low=self.task_params['delay_dur'][0],high=self.task_params['delay_dur'][1],size=(1,))[0]
        delay_end = delay_dur+onset+stim_dur
        inputs[onset:onset+stim_dur,0]=cp
        inputs[onset:onset+stim_dur,1]=sp
        inputs[delay_end:,2]=1
        targets = torch.zeros(self.task_params['trial_len'],2)
        targets[delay_end:,0]=cp
        targets[delay_end:,1]=sp
        mask=torch.zeros_like(targets)
        mask[onset+stim_dur:]=1
        #mask[delay_end:]=5

        return inputs,targets,mask
    

