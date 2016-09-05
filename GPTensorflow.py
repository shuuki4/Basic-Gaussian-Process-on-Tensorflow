import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

sample_trainx = np.asarray([0.1, 0.7, 1.2, 1.4, 1.6, 1.9, 2.4, 2.7, 3.1, 3.5, 3.9, 4.3, 4.8, 5.3], dtype=np.float32)
sample_trainy = np.asarray([2.1, 1.6, 0.5, 0.1, 0.5, 1.2, 2.4, 3.0, 1.6, 2.3, 2.4, 1.8, 1.2, 2.0], dtype=np.float32)
sample_testx = [0.0, 0.3, 1.5, 1.8, 2.6, 3.0, 3.6, 4.2, 5.0, 5.5, 6.0, 6.5]

@ops.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminantGrad(op, grad) :
	A = op.inputs[0]
	C = op.outputs[0]
	Ainv = tf.matrix_inverse(A)
	return grad*C*tf.transpose(Ainv)

def kernel(x, sig_f, sig_n, l, length) :
	# generate kernel matrix
	
	x = x.reshape([-1, 1])
	k = sig_f * sig_f * np.exp(-np.power(x - x.T, 2) / (2*l*l)) + sig_n * sig_n * np.diag(np.ones([length]))
	return k

def kernel_tensor(x, sig_f, sig_n, l, length) :
	# generate kernel matrix by tensor

	x = tf.reshape(x, [-1, 1])
	k = sig_f * sig_f * tf.exp(-tf.pow(x - tf.transpose(x), 2) / (2 * l * l)) + sig_n * sig_n * tf.diag(tf.ones([length]))
	return k

def train(x, y, epoch=200) :
	# train hyperparameters per given train set

	sig_f = tf.Variable(tf.ones([]))
	sig_n = tf.Variable(tf.ones([]))
	l = tf.Variable(tf.ones([]))

	length = x.shape[0]
	x = tf.constant(x)
	y = tf.reshape(tf.constant(y), [-1, 1])
	k = kernel_tensor(x, sig_f, sig_n, l, length)

	loss = 0.5 * (tf.matmul(tf.transpose(y), tf.matmul(tf.matrix_inverse(k), y)) + tf.log(tf.matrix_determinant(k)))
	opt = tf.train.RMSPropOptimizer(learning_rate=0.005, decay=0.9, momentum=0.0).minimize(loss)

	config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
	with tf.Session(config=config) as sess :
		sess.run(tf.initialize_all_variables())
		for epoch in xrange(epoch) :
			_, loss_ = sess.run([opt, loss])
		return sess.run([sig_f, sig_n, l])

def inf(trainx, trainy, testx, hyp) :

	mus = [] ; sigmas = []
	for x in sample_testx :
		xs = np.concatenate((trainx, np.asarray([x]).reshape((1, ))), axis=0)
		length = xs.shape[0]
		y = trainy.reshape((-1, 1))
		k = kernel(xs, hyp[0], hyp[1], hyp[2], length)

		k_star = k[length-1, 0:length-1].reshape((1, -1))
		k_starstar = k[length-1, length-1].reshape((1, 1))
		k = k[0:length-1, 0:length-1]

		mu = np.matmul(np.matmul(k_star, np.linalg.inv(k)), y)
		var = k_starstar - np.matmul(np.matmul(k_star, np.linalg.inv(k)), k_star.T)
		mus.append(mu)
		sigmas.append(var)

	return mus, sigmas

if __name__ == "__main__" :

	hyp = train(sample_trainx, sample_trainy)
	test_mu, test_sig = inf(sample_trainx, sample_trainy, sample_testx, hyp)
	for mu, sig in zip(test_mu, test_sig) :
		print mu, sig
