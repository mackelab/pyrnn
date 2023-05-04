'''
AdaptiveGradNormClip.py
for Python 3.9

Originally written for Python 3.6.9 and TensorFlow 2.8.0
by Matt Golub, October 2018.

Modified by Matthijs Pals
'''
import os
import numpy as np
import pickle

class AdaptiveGradNormClip(object):
	"""Class for managing adaptive gradient norm clipping for stabilizing any gradient-descent-like procedure.

	Essentially, just a running buffer of gradient norms from the last n gradient steps, with a hook into the x-th percentile of those values, which is intended to be used to set the ceiling on the gradient applied at the next iteration of a gradient-descent-like procedure.

	The standard usage is as follows:

	```python
	# Set hyperparameters as desired.
	agnc_hps = dict()
	agnc_hps['sliding_window_len'] = 1.0
	agnc_hps['percentile'] = 95
	agnc_hps['init_clip_val' = 1.0
	agnc_hps['verbose'] = False
	agnc = AdaptiveGradNormClip(**agnc_hps)

	while some_conditions(...):
		# This loop defines one step of the training procedure.

		gradients = get_gradients(data, params)
		grad_norm = compute_gradient_norm(gradients)
		clip_val = agnc.update(grad_norm)
		clipped_gradients = clip_gradients(gradients, clip_val)
		params = apply_gradients(clipped_gradients)

		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveGradNormClip object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			agnc.save(...)
	```

	"""

	''' Included for ready access by RecurrentWhisperer
		(before initializing an instance) '''
	default_hps = {
		'do_adaptive_clipping': True,
		'sliding_window_len': 128,
		'percentile': 95.0,
		'init_clip_val': 1e12,
		'max_clip_val': 1e12,
		'verbose': False
		}

	def __init__(self,
		do_adaptive_clipping=default_hps['do_adaptive_clipping'],
		sliding_window_len=default_hps['sliding_window_len'],
		percentile=default_hps['percentile'],
		init_clip_val=default_hps['init_clip_val'],
		max_clip_val=default_hps['max_clip_val'],
		verbose=default_hps['verbose']):
		'''Builds an AdaptiveGradNormClip object

		Args:
			A set of optional keyword arguments for overriding the default
			values of the following hyperparameters:

			do_adaptive_clipping: A bool indicating whether to implement adaptive gradient norm clipping (i.e., the purpose of this class). Setting to False leads to clipping at a fixed gradient norm specified by fixed_clip_val. Default: True

			sliding_window_len: An int specifying the number of recent steps to
			record. Default: 100.

			percentile: A float between 0.0 and 100.0 specifying the percentile
			of the recorded gradient norms at which to set the clip value.
			Default: 95.

			init_clip_val: A float specifying the initial clip value (i.e., for
			step 1, before any empirical gradient norms have been recorded).
			Default: 1e12.

				This default effectively prevents any clipping on iteration one.
				This has the unfortunate side effect of throwing the vertical
				axis scale on the corresponding Tensorboard plot. The
				alternatives are computationally inefficient: either clip at an
				arbitrary level (or at 0) for the first epoch or compute a
				gradient at step 0 and initialize to the norm of the global
				gradient.

			max_clip_val: A positive float indicating the largest allowable  clipping value. This effectively overrides the adaptive nature of the gradient clipping once the adaptive clip value exceeds this threshold. When do_adaptive_clipping is set to False, this clipping value is always applied at each step. Default: 1e12.

			verbose: A bool indicating whether or not to print status updates.
			Default: False.
		'''
		self.step = 0
		self.do_adaptive_clipping = do_adaptive_clipping
		self.sliding_window_len = sliding_window_len
		self.percentile = percentile
		self.max_clip_val = max_clip_val
		self.grad_norm_log = []
		self.verbose = verbose
		self.save_filename = 'norm_clip.pkl'

		if self.do_adaptive_clipping:
			self.clip_val = init_clip_val
		else:
			self.clip_val = self.max_clip_val

	def __call__(self):
		'''Returns the current clip value.

		Args:
			None.

		Returns:
			A float specifying the current clip value.
		'''
		return self.clip_val

	def update(self, grad_norm):
		'''Update the log of recent gradient norms and the corresponding
		recommended clip value.

		Args:
			grad_norm: A float specifying the gradient norm from the most
			recent gradient step.

		Returns:
			None.
		'''
		if self.do_adaptive_clipping:
			if self.step < self.sliding_window_len:
				# First fill up an entire "window" of values
				self.grad_norm_log.append(grad_norm)
			else:
				# Once the window is full, overwrite the oldest value
				idx = np.mod(self.step, self.sliding_window_len)
				self.grad_norm_log[idx] = grad_norm

			proposed_clip_val = \
				np.percentile(self.grad_norm_log, self.percentile)

			self.clip_val = min(proposed_clip_val, self.max_clip_val)

		self.step += 1

	def save(self, save_dir):
		'''Saves the current AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.

		Args:
			save_dir: A string containing the directory in which to save the
			current object state.

		Returns:
			None.
		'''

		if self.verbose:
			print('Saving AdaptiveGradNormClip.')
		save_path = os.path.join(save_dir, self.save_filename)
		file = open(save_path,'wb')
		file.write(pickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_dir):
		'''Loads a previously saved AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.

		Args:
			restore_dir: A string containing the directory from which to load
			a previously saved object state.

		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveGradNormClip.')
		restore_path = os.path.join(restore_dir, self.save_filename)
		file = open(restore_path,'rb')
		restore_data = file.read()
		file.close()
		self.__dict__ = pickle.loads(restore_data)
