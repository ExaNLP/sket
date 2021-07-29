import numpy as np


class StandardizationNormalizer(object):
	# apply standard deviation normalization
	def __init__(self, scores):
		self.mean = np.mean(scores)
		self.std = np.std(scores)

	def __call__(self, scores):
		if self.std > 0:
			return (scores - self.mean) / self.std
		else:
			return np.zeros(scores.size)


class MinMaxNormalizer(object):
	# apply minmax normalization
	def __init__(self, scores):
		self.min = np.min(scores)
		self.max = np.max(scores)

	def __call__(self, scores):
		if (self.max - self.min) > 0:
			return (scores - self.min) / (self.max - self.min)
		else:
			return np.zeros(scores.size)


class IdentityNormalizer(object):
	# apply identify normalization
	def __init__(self):
		pass

	def __call__(self, scores):
		return scores
