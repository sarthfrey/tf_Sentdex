import tensorflow as tf
import time


x1 = tf.constant(5)
x2 = tf.constant(6)

'''
s1 = time.time()
x1*x2
s2 = time.time()
tf.multiply(x1,x2)
s3 = time.time()
print s2-s1
print s3-s2
'''

result = tf.multiply(x1,x2)

with tf.Session() as sess:
	out = sess.run(result)
	print out

print out # error

