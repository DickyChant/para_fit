

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt



def add_layer(inputs,in_size,out_size,activation_function=None):
	## add one more layer and return the output of this layer
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
		with tf.name_scope('biases'):
			biases=tf.Variable(tf.zeros([5,out_size])+0.1)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b=tf.matmul(inputs,Weights)+biases
	
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	
	return outputs
	
def get_random_block_from_data(data0,mark0):
	num_events = data0.shape[0]
	indices = np.arange(num_events)
	np.random.shuffle(indices)
	data_x = np.zeros((200,5))
	data_y = np.zeros((200,56))
	data_x = data0[indices,:]
	data_y = mark0[indices,:]
	start_indice = np.random.randint(0,190)
	return(data_x[start_indice:start_indice+5,:],data_y[start_indice:start_indice+5,:])	

y_data0 = np.loadtxt("./result/result/result_after_sorted/all02.txt")
x_data0 = np.loadtxt("./result/result/result_after_sorted/input_parameter01.txt")
## define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs=tf.placeholder(tf.float32,[None,5],name='x_input')
	ys=tf.placeholder(tf.float32,[None,56],name='y_input')
	lr=tf.placeholder(tf.float32,name='lr')
##　add hidden layer
l1=add_layer(xs,5,100,activation_function=tf.nn.relu)
## add output layer
prediction=add_layer(l1,100,56,activation_function=None)
prediction1 = tf.reshape(prediction,[-1,1])
with tf.name_scope('loss'):

	loss=tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(ys,[-1,1])-prediction1),reduction_indices=[1]))
with tf.name_scope('train'):
	train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
max_learning_rate = 1e-2
min_learning_rate = 1e-4
decay_speed = 5e4

for i in range(200000):
	learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
	x_data1,y_data1 = get_random_block_from_data(x_data0,y_data0)
	sess.run(train_step,feed_dict={xs:x_data1,ys:y_data1,lr:learning_rate})
	
	if i % 1000==0:
		print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1,lr:learning_rate}))
		#print(sess.run(ys,feed_dict={xs:x_data1,ys:y_data1}))

y_data3 = sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1,lr:learning_rate})
fig=plt.figure() ##加一个图片框
ax=fig.add_subplot(1,1,1)
x_data3 = np.linspace(-5,5,280)[:,np.newaxis]
ax.scatter(x_data3,y_data3.reshape(-1,1))
plt.ion()
plt.show()
lines=ax.plot(x_data3,y_data1.reshape(-1,1),'r-',lw=1)
plt.pause(30)
