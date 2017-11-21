"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import numpy
import scipy

"""
This is a python implementation of vanilla lda, based on a lda approach of variational inference and gibbs sampling, with hyper parameter updating.
It supports asymmetric Dirichlet prior over the topic simplex.
"""


def compute_dirichlet_expectation(dirichlet_parameter):
	if (len(dirichlet_parameter.shape) == 1):
		return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
	return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:,
	                                                numpy.newaxis]


def parse_vocabulary(vocab):
	type_to_index = {}
	index_to_type = {}
	for word in set(vocab):
		index_to_type[len(index_to_type)] = word
		type_to_index[word] = len(type_to_index)

	return type_to_index, index_to_type


class Inferencer():
	"""
	"""

	def __init__(self,
	             # local_parameter_iterations=50,
	             hyper_parameter_optimize_interval=10,
	             ):

		self._hyper_parameter_optimize_interval = hyper_parameter_optimize_interval

	# self._local_parameter_iterations = local_parameter_iterations
	# assert(self._local_maximum_iteration>0)

	"""
	"""

	def _initialize(self, vocab, number_of_topics, alpha_mu, alpha_sigma, beta_mu, beta_sigma):
		self.parse_vocabulary(vocab)

		# initialize the size of the vocabulary, i.e. total number of distinct tokens.
		self._number_of_types = len(self._type_to_index)

		self._counter = 0

		# initialize the total number of topics.
		self._number_of_topics = number_of_topics

		# clock = time.clock()
		# initialize a K-dimensional vector, valued at 1/K.
		if self._diagonal_covariance_matrix:
			self._alpha_mu = numpy.zeros(self._number_of_topics) + alpha_mu
			self._alpha_sigma = numpy.ones(self._number_of_topics) + alpha_sigma
		else:
			self._alpha_mu = numpy.zeros((1, self._number_of_topics)) + alpha_mu
			self._alpha_sigma = numpy.eye(self._number_of_topics) * alpha_sigma
			self._alpha_sigma_inv = numpy.eye(self._number_of_topics) / alpha_sigma
		# self._alpha_sigma_inv = numpy.linalg.pinv(self._alpha_sigma)
		# print "check point 1", time.clock() - clock

		# clock = time.clock()
		if self._diagonal_covariance_matrix:
			self._beta_mu = numpy.zeros(self._number_of_types) + beta_mu
			self._beta_sigma = numpy.ones(self._number_of_types) + beta_sigma
		else:
			self._beta_mu = numpy.zeros((1, self._number_of_types)) + beta_mu

			self._beta_sigma = numpy.eye(self._number_of_types) * beta_sigma
			self._beta_sigma_inv = numpy.eye(self._number_of_types) / beta_sigma
		# self._beta_sigma_inv = numpy.linalg.pinv(self._beta_sigma)
		# print "check point 2", time.clock() - clock

	def parse_vocabulary(self, vocab):
		self._type_to_index = {}
		self._index_to_type = {}
		for word in set(vocab):
			self._index_to_type[len(self._index_to_type)] = word
			self._type_to_index[word] = len(self._type_to_index)

		self._vocab = self._type_to_index.keys()

	def parse_data(self):
		raise NotImplementedError

	"""
	"""

	def learning(self):
		raise NotImplementedError

	"""
	"""

	def inference(self):
		raise NotImplementedError

	def export_beta(self, exp_beta_path, top_display=-1):
		raise NotImplementedError


if __name__ == "__main__":
	raise NotImplementedError
