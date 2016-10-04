import tensorflow as tf




#input -> weight ->> hidden layer1 2 -> weights -> output layer

#compare output to inteded output -> cost func
#optilization func -> min cost
#backprop
#feed forward + backprop = epoch

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#one_hot = one is on rest is off
# 10 class, 0-9
#one hot makes binary 
#ex 0  = [1,0,0,0,0,0,0,0,0,0,0,]
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_class = 10

batch_size=100
#set of batch is a small bucket to store data


x = tf.placeholder("float",[None, 784])
#heightx width it shows error when it goes over this limit
y = tf.placeholder("float")

def  neural_network_model(data):


	#(input_data * weights) + bias
	#weights = strength between two nodes
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
		'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} 

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
		'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))} 

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
		'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))} 

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_class])),
		'biases':tf.Variable(tf.random_normal([n_class]))} 

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) , hidden_1_layer['biases']) 
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']) ,hidden_2_layer['biases']) 
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']) , hidden_3_layer['biases']) 
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#minimize
	#learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#cycle feed forward and back prop
	hm_epochs = 10


	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			#it tells us the needed how many times we need to cycle
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x,y: epoch_y})
				epoch_loss += c
			print('Epoch',epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)

			#argmax = return index of the max value 
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



train_neural_network(x)


'''
	In machine-learning parlance, an epoch is a complete pass through a given dataset.
	 That is, by the end of one epoch, your neural network – be it a restricted Boltzmann machine,
	  convolutional net or deep-belief network – will have been exposed to every record to example within 
	  the dataset once. Not to be confused with an iteration, which is simply one update of
	   the neural net model’s parameters. Many iterations can occur before an epoch is over.
	    Epoch and iteration are only synonymous if you update your parameters once for each 
	    pass through the whole dataset; if you update using mini-batches, they mean different things.
	     Say your data has 2 minibatches: A and B. .numIterations(3) performs training like AAABBB, 
	     while 3 epochs looks like ABABAB.
'''






