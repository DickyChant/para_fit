import tensorflow as tf
import numpy as np
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr,formataddr
from email import encoders
import smtplib


def add_layer(inputs,in_size,out_size,activation_function=None,if_b=True):
	## add one more layer and return the output of this layer
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights=tf.Variable(10*tf.random_normal([in_size,out_size]),name='W')
		with tf.name_scope('biases'):
			biases=tf.Variable(tf.zeros([40,out_size])+0.1)
		with tf.name_scope('Wx_plus_b'):
			if if_b:
				Wx_plus_b=tf.matmul(inputs,Weights)+biases
			else:
				Wx_plus_b=tf.matmul(inputs,Weights)
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)

	return outputs,Weights,biases

def add_conv(xs,dim_in,activation_function=None):##默认32层卷积核，我没写太复杂233333
	initial=tf.truncated_normal(shape=[1,5,dim_in,32],stddev=0.1)
	W=tf.Variable(initial)
	#xs=np.reshape(xs,[40,None])
	xs1=tf.expand_dims(xs,0)
	xs1=tf.expand_dims(xs1,0)
	conv=tf.nn.conv2d(xs1,W,strides=[1,1,1,1],padding='SAME')
	conv=tf.squeeze(conv)
	if activation_function is None:
		outputs=conv
	else:
		outputs=activation_function(conv)

	return outputs


def get_random_block_from_data(data0,mark0):
	num_events = data0.shape[0]
	indices = np.arange(num_events)
	np.random.shuffle(indices)
	data_x = np.zeros((200,56))
	data_y = np.zeros((200,5))
	data_x = data0[indices,:]
	data_y = mark0[indices,:]
	start_indice = np.random.randint(0,176)
	return(data_x[start_indice:start_indice+44,:],data_y[start_indice:start_indice+44,:])

def get_ordered_block_from_data(data0,mark0):
	data_x=data0.reshape([5,44,56])
	data_y=mark0.reshape([5,44,5])
	return(data_x,data_y)
x_data0 = np.loadtxt("./obserables_for_train.txt ")
#x_data0=x_data0*10
y_data0 = np.loadtxt("./parameters_for_train.txt")
#y_data0[:,2]*=20
## define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs=tf.placeholder(tf.float32,[None,56],name='x_input')
	ys=tf.placeholder(tf.float32,[None,5],name='y_input')
##　add hidden layer
l1=tf.layers.dense(xs,64,activation=tf.nn.relu,use_bias=False)
l1=tf.layers.dropout(l1)
l2=tf.layers.dense(l1,128,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
l2=tf.layers.dense(l2,256,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
l2=tf.layers.dense(l2,512,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
l2=tf.layers.dense(l2,256,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
l2=tf.layers.dense(l2,128,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
l2=tf.layers.dense(l2,64,activation=tf.nn.relu,use_bias=False)
l2=tf.layers.dropout(l2)
## add conv layer
#l1=add_conv(l1,100,activation_function=tf.nn.relu)
## add output layer
prediction=tf.layers.dense(l2,5,use_bias=False)
prediction1 = tf.reshape(prediction,[-1,1])
with tf.name_scope('loss'):

	loss=tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(ys,[-1,1])-prediction1),reduction_indices=[1]))
with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(0.0005).minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)


for i in range(500000):

	x_data1, y_data1 = get_random_block_from_data(x_data0, y_data0)
		#y_data1=y_data1*10
	sess.run(train_step,feed_dict={xs:x_data1,ys:y_data1})

	if i % 10000==0:
		print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}),i)
		#print(sess.run(ys,feed_dict={xs:x_data1,ys:y_data1}))
#y_data0[:,2]/=20
x_data1,y_data1 = get_random_block_from_data(x_data0,y_data0)
y_data1=y_data1
pred=sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1})
print(pred)
print(y_data1)
print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}))
x_data1,y_data1 = get_random_block_from_data(x_data0,y_data0)
y_data1=y_data1
##pred=sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1})
print(sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1}))
print(y_data1)
print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}))

x_test=np.loadtxt('./obserables_for_test.txt')
y_test=np.loadtxt('./parameters_for_test.txt')
#y_test[:,2]*=20
x_test=np.tile(x_test,[2,1])
y_test=np.tile(y_test,[2,1])

x_test1=x_test[0:44,:]
y_test1=y_test[0:44,:]

print(sess.run(prediction,feed_dict={xs:x_test1,ys:y_test1}))
print(y_test1)
print(sess.run(loss,feed_dict={xs:x_test1,ys:y_test1}))
a=sess.run(prediction,feed_dict={xs:x_test1,ys:y_test1})
#a[:,2]/=20
#y_test1[:,2]/=20
print(y_test1[0:23,:])
b=(a-y_test1)/y_test1
c=np.zeros([23,5,2])
c[:,:,0]=y_test1[0:23,:]
#y_test1[:,2]*=20
c[:,:,1]=b[0:23,:]
np.savetxt('./ob_to_p/predictions.txt',a[0:23,:])
np.savetxt('./ob_to_p/error_relative.txt',b[0:23,:])
np.set_printoptions(precision=3, suppress=True)
print(b[0:23,:])