'''
AdaptiveLearningRate.py
for Python 3.9

Originally written for Python 3.6.9 and TensorFlow 2.8.0
by Matt Golub, October 2018.

Modified by Matthijs Pals
'''
import os
import pickle
import numpy as np
import numpy.random as npr

if os.environ.get('DISPLAY','') == '':
	# Ensures smooth running across environments, including servers without
	# graphical backends.
	print('No display found. Using non-interactive Agg backend.')
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AdaptiveLearningRate(object):
	'''Class for managing an adaptive learning rate schedule based on the
	recent history of loss values. The adaptive schedule begins with an
	optional warm-up period, during which the learning rate ramps up (with
	user-specified shape) to the initial rate. For the remainder of the training
	procedure, the learning rate will increase following a period of monotonic
	improvements in the loss and will decrease if a loss is encountered that
	is worse than all losses in the recent period. Hyperparameters control the
	length of each of these periods and the extent of each type of learning
	rate change.

	Note that this control flow is asymmetric--stricter criteria must be met
	for increases than for decreases in the learning rate This choice 1)
	encourages decreases in the learning rate when moving into regimes with a
	flat loss surface, and 2) attempts to avoid instabilities that can arise
	when the learning rate is too high (and the often irreversible
	pathological parameter updates that can result). Practically,
	hyperparameters may need to be tuned to optimize the learning schedule and
	to ensure that the learning rate does not explode.

	See test(...) to simulate learning rate trajectories based on specified
	hyperparameters.

	The standard usage is as follows:

	# Set hyperparameters as desired.
	alr_hps = dict()
	alr_hps['initial_rate'] = 1.0
	alr_hps['min_rate'] = 1e-3
	alr_hps['max_n_steps'] = 1e4
	alr_hps['n_warmup_steps'] = 0
	alr_hps['warmup_scale'] = 1e-3
	alr_hps['warmup_shape'] = 'gaussian'
	alr_hps['do_decrease_rate'] = True
	alr_hps['min_steps_per_decrease'] = 5
	alr_hps['decrease_factor'] = 0.95
	alr_hps['do_increase_rate'] = True
	alr_hps['min_steps_per_increase'] = 5
	alr_hps['increase_factor'] = 1./0.95
	alr_hps['verbose'] = False
	alr = AdaptiveLearningRate(**alr_hps)

	# This loop iterates through the optimization procedure.
	while ~alr.is_finished():
		# Get the current learning rate
		learning_rate = alr()

		# Use the current learning rate to update the model parameters.
		# Get the loss of the model after the update.
		params, loss = run_one_training_step(params, learning_rate, ...)

		# Update the learning rate based on the most recent loss value
		# and an internally managed history of loss values.
		alr.update(loss)

		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveLearningRate object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			alr.save(...)
	'''

	''' Included for ready access by RecurrentWhisperer
		(before initializing an instance) '''
	default_hps = {
		'initial_rate': 1.0,
		'min_rate': 1e-3,
		'max_n_steps': 1e4,
		'n_warmup_steps': 0,
		'warmup_scale': 1e-3,
		'warmup_shape': 'gaussian',
		'do_decrease_rate': True,
		'min_steps_per_decrease': 5,
		'decrease_factor': 0.95,
		'do_increase_rate': True,
		'min_steps_per_increase': 5,
		'increase_factor': 1/0.95,
		'verbose': False
		}

	def __init__(self,
		initial_rate = default_hps['initial_rate'],
		min_rate = default_hps['min_rate'],
		max_n_steps = default_hps['max_n_steps'],
		n_warmup_steps = default_hps['n_warmup_steps'],
		warmup_scale = default_hps['warmup_scale'],
		warmup_shape = default_hps['warmup_shape'],
		do_decrease_rate = default_hps['do_decrease_rate'],
		min_steps_per_decrease = default_hps['min_steps_per_decrease'],
		decrease_factor = default_hps['decrease_factor'],
		do_increase_rate = default_hps['do_increase_rate'],
		min_steps_per_increase = default_hps['min_steps_per_increase'],
		increase_factor = default_hps['increase_factor'],
		verbose = default_hps['verbose']):
		'''Builds an AdaptiveLearningRate object

		Args:
			A set of optional keyword arguments for overriding the default
			values of the following hyperparameters:

			initial_rate: Non-negative float specifying the initial learning
			rate. Default: 1.0.

			min_rate: Non-negative float specifying the largest learning
			rate for which is_finished() returns False. This can optionally be
			used externally to signal termination of the optimization
			procedure. This argument is never used internally--the learning
			rate behavior doesn't depend on this value. Default: 1e-3.

			max_n_steps: Non-negative integer specifying the maximum number of
			steps before is_finished() will return True. This can optionally be
			used externally to signal termination of the optimization
			procedure. This argument is never used internally--the learning
			rate behavior doesn't depend on this value. Default: 1e4.

			n_warmup_steps: Non-negative int specifying the number of warm-up
			steps to take. During these warm-up steps, the learning rate will
			monotonically increase up to initial_rate (according to
			warmup_scale and warmup_shape). Default: 0 (i.e., no
			warm-up).

			warmup_scale: Float between 0 and 1 specifying the learning rate
			on the first warm-up step, relative to initial_rate. The first
			warm-up learning rate is warmup_scale * initial_rate. Default:
			0.001.

			warmup_shape: 'linear', 'exp', or 'gaussian', indicating the shape
			of the increasing learning rate during the warm-up period.
			Default: 'gaussian'.

			do_decrease_rate: Bool indicating whether or not to decrease the
			learning rate during training (after any warm-up). Default: True.

			min_steps_per_decrease: Non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			decrease the learning rate. Learning rate decreases are made when
			a loss value is encountered that is worse than every loss value in
			this window. When the learning rate is decreased, no further
			decreases are considered until this many new steps have
			transpired. Larger values will slow convergence due to the
			learning rate. Default 5.

			decrease_factor: Float between 0 and 1 specifying the extent of
			learning rate decreases. Whenever a decrease is made, the learning
			rate decreases from x to decrease_factor * x. Values closer to 1
			will slow convergence due to the learning rate. Default: 0.95.

			do_increase_rate: Bool indicating whether or not to increase the
			learning rate during training (after any warm-up). Default: True.

			min_steps_per_increase: Non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			increase the learning rate. Learning rate increases are made when
			the loss has monotonically decreased over this many steps. When
			the learning rate is increased, no further increases are
			considered until this many new steps have transpired. Default 5.

			increase_factor: Float greater than 1 specifying the extent of
			learning rate increases. Whenever an increase is made, the
			learning rate increases from x to increase_factor * x. Larger
			values will slow convergence due to the learning rate. Default:
			1./0.95.

			verbose: Bool indicating whether or not to print status updates.
			Default: False.
		'''

		self.step = 0
		self.step_last_update = -1
		self.prev_rate = None
		self.loss_log = []

		self.initial_rate = initial_rate
		self.min_rate = min_rate
		self.max_n_steps = max_n_steps
		self.do_decrease_rate = do_decrease_rate
		self.decrease_factor = decrease_factor
		self.min_steps_per_decrease = min_steps_per_decrease
		self.do_increase_rate = do_increase_rate
		self.increase_factor = increase_factor
		self.min_steps_per_increase = min_steps_per_increase

		self.n_warmup_steps = n_warmup_steps
		self.warmup_scale = warmup_scale
		self.warmup_shape = warmup_shape.lower()

		self.save_filename = 'learning_rate.pkl'

		self._validate_hyperparameters()

		self.warmup_rates = self._get_warmup_rates()

		self.verbose = verbose

		if n_warmup_steps > 0:
			self.learning_rate = self.warmup_rates[0]
		else:
			self.learning_rate = initial_rate

		if self.verbose:
			print('AdaptiveLearningRate schedule requires at least %s steps:' %
				str(self.min_steps))

	def __call__(self):
		'''Returns the current learning rate.'''

		return self.learning_rate

	def is_finished(self, do_check_step=True, do_check_rate=True):
		''' Indicates termination of the optimization procedure. Note: this
		function is never used internally and does not influence the behavior
		of the adaptive learning rate.

		Args:
			do_check_step: Bool indicating whether to check if the step has
			reached max_n_steps.

			do_check_rate: Bool indicating whether to check if the learning rate
			has fallen below min_rate.

		Returns:
			Bool indicating whether any of the termination criteria have been
			met.
		'''

		if do_check_step and self.step > self.max_n_steps:
			return True
		elif self.step <= self.n_warmup_steps:
			return False
		elif do_check_rate and self.learning_rate <= self.min_rate:
			return True
		else:
			return False

	@property
	def min_steps(self):
		''' Computes the minimum number of steps required before the learning
		rate falls below the min_rate, i.e., assuming the rate decreases at
		every opportunity permitted by the properties of this
		AdaptiveLearningRate object.

		Args:
			None.

		Returns:
			An int specifying the minimum number of steps in the adaptive
			learning rate schedule.
		'''
		n_decreases = np.ceil(np.divide(
			(np.log(self.min_rate) - np.log(self.initial_rate)),
			np.log(self.decrease_factor)))
		return self.n_warmup_steps + self.min_steps_per_decrease * n_decreases

	def update(self, loss):
		'''Updates the learning rate based on the most recent loss value
		relative to the recent history of loss values.

		Args:
			loss: A float indicating the loss from the current training step.

		Returns:
			A float indicating the updated learning rate.
		'''
		self.loss_log.append(loss)

		step = self.step
		cur_rate = self.learning_rate
		step_last_update = self.step_last_update

		self.prev_rate = cur_rate

		if step <= self.n_warmup_steps:
			'''If step indicates that we are still in the warm-up, the new rate is determined entirely based on the warm-up schedule.'''
			if step < self.n_warmup_steps:
				self.learning_rate = self.warmup_rates[step]
				if self.verbose:
					print('Warm-up (%d of %d): Learning rate set to %.2e'
						  % (step+1,self.n_warmup_steps,self.learning_rate))
			else: # step == n_warmup_steps:
				self.learning_rate = self.initial_rate
				if self.verbose:
					print('Warm-up complete (or no warm-up). Learning rate set to %.2e'
						  % self.learning_rate)
			self.step_last_update = step

			'''Otherwise, rate may be kept, increased, or decreased based on
			recent loss history.'''
		elif self._conditional_decrease_rate():
			self.step_last_update = step
		elif self._conditional_increase_rate():
			self.step_last_update = step

		self.step = step + 1

		return self.learning_rate

	def save(self, save_dir):
		'''Saves the current state of the AdaptiveLearningRate object.

		Args:
			save_dir: A string containing the directory in which to save.

		Returns:
			None.
		'''
		if self.verbose:
			print('Saving AdaptiveLearningRate.')
		save_path = os.path.join(save_dir, self.save_filename)
		file = open(save_path,'wb')
		file.write(pickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_dir):
		'''Restores the state of a previously saved AdaptiveLearningRate
		object.

		Args:
			restore_dir: A string containing the directory in which to find a
			previously saved AdaptiveLearningRate object.

		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveLearningRate.')
		restore_path = os.path.join(restore_dir, self.save_filename)
		file = open(restore_path,'rb')
		restore_data = file.read()
		file.close()
		self.__dict__ = pickle.loads(restore_data)

	def _validate_hyperparameters(self):
		'''Checks that critical hyperparameters have valid values.

		Args:
			None.

		Returns:
			None.

		Raises:
			Various ValueErrors depending on the violating hyperparameter(s).
		'''
		def assert_non_negative(attr_str):
			'''
			Args:
				attr_str: The name of a class variable.

			Returns:
				None.

			Raises:
				ValueError('%s must be non-negative but was %d' % (...))
			'''
			val = getattr(self, attr_str)
			if val < 0:
				raise ValueError('%s must be non-negative but was %d'
								 % (attr_str, val))

		assert_non_negative('initial_rate')
		assert_non_negative('n_warmup_steps')
		assert_non_negative('min_steps_per_decrease')
		assert_non_negative('min_steps_per_increase')

		if self.decrease_factor > 1.0 or self.decrease_factor < 0.:
			raise ValueError('decrease_factor must be between 0 and 1, '
			                 'but was %f' % self.decrease_factor)

		if self.increase_factor < 1.0:
			raise ValueError('increase_factor must be >= 1, but was %f'
							 % self.increase_factor)

		if self.warmup_shape not in ['exp', 'gaussian', 'linear']:
			raise ValueError('warmup_shape must be \'exp\' or \'gaussian\', '
			                 'but was %s' % self.warmup_shape)

	def _get_warmup_rates(self):
		'''Determines the warm-up schedule of learning rates, culminating at
		the desired initial rate.

		Args:
			None.

		Returns:
			Shape (n_warmup_steps,) numpy array containing the learning rates
			for each step of the warm-up period.

		'''
		n = self.n_warmup_steps
		warmup_shape = self.warmup_shape
		scale = self.warmup_scale
		warmup_start = scale*self.initial_rate
		warmup_stop = self.initial_rate

		if warmup_shape == 'linear':
			warmup_rates = np.linspace(warmup_start, warmup_stop, n+1)[:-1]
		if self.warmup_shape == 'exp':
			warmup_rates = np.logspace(
				np.log10(warmup_start), np.log10(warmup_stop), n+1)[:-1]
		elif self.warmup_shape == 'gaussian':
			mu = np.float32(n)
			x = np.arange(mu)

			# solve for sigma s.t. warmup_rates[0] = warmup_start
			sigma = np.sqrt(-mu**2.0 / (2.0*np.log(warmup_start/warmup_stop)))

			warmup_rates = warmup_stop*np.exp((-(x-mu)**2.0)/(2.0*sigma**2.0))

		return warmup_rates

	def _conditional_increase_rate(self):
		'''Increases the learning rate if loss values have monotonically
		decreased over the past n steps, and if no learning rate changes have
		been made in the last n steps, where n=min_steps_per_increase.

		Args:
			None.

		Returns:
			A bool indicating whether the learning rate was increased.
		'''

		did_increase_rate = False
		n = self.min_steps_per_increase

		if self.do_increase_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.less(batch_loss_window[1:],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.increase_factor
				did_increase_rate = True
				if self.verbose:
					print('Learning rate increased to %.2e'
						  % self.learning_rate)

		return did_increase_rate

	def _conditional_decrease_rate(self):
		'''Decreases the learning rate if the most recent loss is worse than
		all of the previous n loss values, and if no learning rate changes
		have been made in the last n steps, where n=min_steps_per_decrease.

		Args:
			None.

		Returns:
			A bool indicating whether the learning rate was decreased.
		'''

		did_decrease_rate = False
		n = self.min_steps_per_decrease

		if self.do_decrease_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.greater(batch_loss_window[-1],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.decrease_factor
				did_decrease_rate = True
				if self.verbose:
					print('Learning rate decreased to %.2e'
						  % self.learning_rate)

		return did_decrease_rate

	def test(self, bias=0.0, fig=None):
		''' Generates and plots an adaptive learning rate schedule based on a
		loss function that is a 1-dimensional biased random walk. This can be
		used as a zero-th order analysis of hyperparameter settings,
		understanding that in a realistic optimization setting, the loss will
		depend highly on the learning rate (such dependencies are not included
		in this simulation).

		Args:
			bias: A float specifying the bias of the random walk used to
			simulate loss values.

		Returns:
			None.
		'''

		save_step = min(1000, self.max_n_steps/4)
		save_dir = '/tmp/'

		# Simulation 1
		loss = 0.
		loss_history = []
		rate_history = []
		while not self.is_finished():
			if self.step == save_step:
				print('Step %d: saving so that we can test restore()' %
					self.step)
				self.save(save_dir)

			loss = loss + bias + npr.randn()
			rate_history.append(self.update(loss))
			loss_history.append(loss)

			if np.mod(self.step, 100) == 0:
				print('Step %d...' % self.step)

		print('Step %d: simulation 1 complete.' % self.step)

		# Simlulation 2, tests restore(...)
		restored_alr = AdaptiveLearningRate()
		restored_alr.restore(save_dir)
		restored_rate_history = [np.nan] * save_step
		while not restored_alr.is_finished():
			# Use exactly the same loss values from the first simulation
			loss = loss_history[restored_alr.step]
			restored_rate_history.append(restored_alr.update(loss))

		print('Step %d: simulation 2 complete.' % restored_alr.step)

		diff = np.array(rate_history[save_step:]) - \
			np.array(restored_rate_history[save_step:])
		mean_abs_restore_error = np.mean(np.abs(diff))
		print('Avg abs diff between original and restored: %.3e' %
			mean_abs_restore_error)

		if fig is None:
			fig = plt.figure()

		ax1 = fig.add_subplot(2,1,1)
		ax1.plot(rate_history)
		ax1.plot(restored_rate_history, linestyle='--')
		ax1.set_yscale('log')
		ax1.set_ylabel('Learning rate')

		ax2 = fig.add_subplot(2,1,2)
		ax2.plot(loss_history)
		ax2.set_ylabel('Simulated loss')
		ax2.set_xlabel('Step')

		fig.show()
