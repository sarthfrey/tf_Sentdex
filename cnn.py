import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 10

height = 28
width =28
n_classes = 10
filter_size = 3
batch_size = 100

# height by width
x = tf.placeholder('float',[None, height * width])
y = tf.placeholder('float',[None, n_classes])

def model(data):

	batches = tf.shape(data)[0]

	hl_1 = {'weights': tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, 32], stddev=0.1)),
			'biases': tf.Variable(tf.constant(0.1, shape=[height, width, 32]))}

	hl_2 = {'weights': tf.Variable(tf.truncated_normal([filter_size, filter_size, 32, 64], stddev=0.1)),
			'biases': tf.Variable(tf.constant(0.1, shape=[height/2,width/2, 64]))}

	output = {'weights': tf.Variable(tf.random_normal([(height/4)*(width/4)*64, n_classes])),
			'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.nn.relu(
		tf.nn.max_pool(
			tf.add(
				tf.nn.conv2d(
					tf.reshape(
						data,
						[batches, height, width, 1]
					),
					hl_1['weights'],
					[1,1,1,1], 
					'SAME'
				),
				hl_1['biases']
			),
		[1,2,2,1],
		[1,2,2,1],
		'SAME'
		)
	)

	l2 = tf.nn.relu(
		tf.nn.max_pool(
			tf.add(
				tf.nn.conv2d(
					l1,
					hl_2['weights'],
					[1,1,1,1],
					'SAME'
				),
				hl_2['biases']
			),
		[1,2,2,1],
		[1,2,2,1],
		'SAME'
		)
	)

	out = tf.add(
		tf.matmul(
			tf.reshape(
				l2,
				[batches, (height/4)*(width/4)*64]
			),
			output['weights']
		),
		output['biases']
	)

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


