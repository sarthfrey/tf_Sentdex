import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 10

n_pixels = 784
n_classes = 10
batch_size = 100

# height by width
x = tf.placeholder('float',[None, n_pixels])
y = tf.placeholder('float',[None, n_classes])

def model(data):

	hl_1 = {'weights': tf.Variable(tf.random_normal([n_pixels,n_nodes_hl1])),
			'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hl_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
			'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hl_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
			'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_classes])), #todo
			  'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.nn.relu(tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases']))

	l2 = tf.nn.relu(tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases']))

	l3 = tf.nn.relu(tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases']))

	out = tf.add(tf.matmul(l2, output['weights']), output['biases']) #todo

	return out

def train_model(x):

	prediction = model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print 'Epoch: {0}, completed out of {1} epochs, loss: {2}'.format(epoch, num_epochs, epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy: {0}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_model(x)


