import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr,formataddr
from email import encoders
import smtplib


x_test=np.loadtxt('./obserables_for_test.txt')
y_test=np.loadtxt('./parameters_for_test.txt')

x_data0 = np.loadtxt("./obserables_for_train.txt ")
#x_data0=x_data0*10
y_data0 = np.loadtxt("./parameters_for_train.txt")
#y_data0[:,2]*=20

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

with tf.name_scope('inputs'):
	xs=tf.placeholder(tf.float32,[None,56],name='x_input')
	ys=tf.placeholder(tf.float32,[None,5],name='y_input')

l1=tf.layers.dense(ys,128,activation=tf.nn.relu)
l1=tf.layers.dropout(l1)
l1=tf.layers.dense(l1,128,activation=tf.nn.relu)
l1=tf.layers.dropout(l1)
l1=tf.layers.dense(l1,128,activation=tf.nn.relu)
l1=tf.layers.dropout(l1)
l1=tf.layers.dense(l1,128,activation=tf.nn.relu)
l1=tf.layers.dropout(l1)

output=tf.layers.dense(l1,56)

pred=tf.reshape(output,[-1,1])

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(tf.reshape(xs, [-1, 1]) - pred))

with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(0.0005).minimize(loss)

sess=tf.Session()

init=tf.initialize_all_variables()

sess.run(init)

for i in range(200000):
    x_data1,y_data1=get_random_block_from_data(x_data0,y_data0)
    sess.run(train_step,feed_dict={xs:x_data1,ys:y_data1})

    if i%10000==0:
        print(i,sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}))

x_test=np.tile(x_test,[2,1])
y_test=np.tile(y_test,[2,1])

x_test1=x_test[0:44,:]
y_test1=y_test[0:44,:]

print(sess.run(output,feed_dict={xs:x_test1,ys:y_test1}))
print(y_test1)
print(sess.run(loss,feed_dict={xs:x_test1,ys:y_test1}))
a=sess.run(output,feed_dict={xs:x_test1,ys:y_test1})
#a[:,2]/=20
#y_test1[:,2]/=20
print(x_test1[0:23,:])
b=(a-x_test1)/x_test1
c=np.zeros([23,56,2])
c[:,:,0]=x_test1[0:23,:]
#y_test1[:,2]*=20
c[:,:,1]=b[0:23,:]

#np.savetxt('./p_to_ob/predictions.txt',a[0:23,:])
#np.savetxt('./p_to_ob/error_relative.txt',b[0:23,:])
np.set_printoptions(precision=3, suppress=True)
print(b[0:23,:])



np.set_printoptions(precision=3,suppress=True)


y_data3 = sess.run(output,feed_dict={xs:x_test1,ys:y_test1})
y_data3=y_data3[0:23,:]
fig1=plt.figure(1, dpi=300) ##加一个图片框

#ax=fig.add_subplot(1,1,1)
x_data3 = np.linspace(-5,5,56*23)[:,np.newaxis]
#ax.scatter(x_data3,y_data3.reshape(-1,1))
for i in range(4):
	if i!=0:
		aaa=fig1.add_subplot(2,2,i)
		aaa.scatter(x_data3[(i-1)*6*56:i*6*56],y_data3.reshape(-1,1)[(i-1)*6*56:i*56*6],marker='x',s=10)
		aaa.plot(x_data3[(i-1)*6*56:i*6*56], x_test1[0:23, :].reshape(-1, 1)[(i-1)*6*56:i*6*56], 'r-', lw=0.5)
		fig2=plt.figure(i+1)
		ax=fig2.add_subplot(1,1,1)
		ax.scatter(x_data3[(i-1)*6*56:i*6*56],y_data3.reshape(-1,1)[(i-1)*56*6:i*56*6])
		ax.plot(x_data3[(i-1)*6*56:i*6*56], x_test1[0:23, :].reshape(-1, 1)[(i-1)*6*56:i*6*56], 'r-', lw=0.5)
	else:
		aaa = fig1.add_subplot(2, 2, 4)
		aaa.scatter(x_data3[18*56:], y_data3.reshape(-1, 1)[18*56:],marker='x',s=20)
		aaa.plot(x_data3[18*56:], x_test1[0:23, :].reshape(-1, 1)[18*56:], 'r-', lw=0.5)
		fig2 = plt.figure(5)
		ax = fig2.add_subplot(1, 1, 1)
		ax.scatter(x_data3[18*56:], y_data3.reshape(-1, 1)[18*56:])
		ax.plot(x_data3[18*56:], x_test1[0:23, :].reshape(-1, 1)[18 * 56:], 'r-', lw=0.5)
#lines=ax.plot(x_data3,x_test1[0:23,:].reshape(-1,1),'r-',lw=0.5)
plt.ion()
plt.show()