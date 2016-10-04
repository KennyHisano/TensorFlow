import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)


#multiplication
result = tf.mul(x1,x2)

#print (result)





with tf.Session() as sess:
	output = sess.run(result)
	print(output)


#you can access variable in with
print(output)
