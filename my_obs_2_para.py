
import tensorflow as tf
import numpy as np



def add_layer(inputs ,in_size ,out_size ,activation_function=None):
    ## add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weight s =tf.Variable(tf.random_normal([in_size ,out_size]) ,name='W')
        with tf.name_scope('biases'):
            biase s =tf.Variable(tf.zeros([5 ,out_size] ) +0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_ b =tf.matmul(inputs ,Weights ) +biases

    if activation_function is None:
        output s =Wx_plus_b
    else:
        output s =activation_function(Wx_plus_b)

    return outputs

def get_random_block_from_data(data0 ,mark0):
    num_events = data0.shape[0]
    indices = np.arange(num_events)
    np.random.shuffle(indices)
    data_x = np.zeros((200 ,4))
    data_y = np.zeros((200 ,5))
    data_x = data0[indices ,:]
    data_y = mark0[indices ,:]
    start_indice = np.random.randint(0 ,195)
    retur n(data_x[start_indice:start_indic e +5 ,:] ,data_y[start_indice:start_indic e +5 ,:])

x_data0 = np.loadtxt("./result/result/result_after_sorted/all04.txt")
y_data0 = np.loadtxt("./result/result/result_after_sorted/input_parameter01.txt")
## define placeholder for inputs to network
with tf.name_scope('inputs'):
    x s =tf.placeholder(tf.float32 ,[None ,4] ,name='x_input')
    y s =tf.placeholder(tf.float32 ,[None ,5] ,name='y_input')
##　add hidden layer
l1 =add_layer(xs ,4 ,100 ,activation_function=tf.nn.relu)
on=tf.nn.relu)##　add hidden layer
l1 =add_layer(l1 ,100 ,100 ,activation_functi
## add output layer
predictio n =add_layer(l1 ,100 ,5 ,activation_function=None)
prediction1 = tf.reshape(prediction ,[-1 ,1])
with tf.name_scope('loss'):

    los s =tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(ys ,[-1 ,1] ) -prediction1) ,reduction_indices=[1]))
with tf.name_scope('train'):
    train_step_ 1 =tf.train.AdamOptimizer(0.0001).minimize(loss)
    train_step_ 2 =tf.train.GradientDescentOptimizer(0.005).minimize(loss)
ini t =tf.initialize_all_variables()

ses s =tf.Session()
write r =tf.summary.FileWriter("logs/" ,sess.graph)
sess.run(init)

for i in range(200000):
    x_data1 ,y_data1 = get_random_block_from_data(x_data0 ,y_data0)
    sess.run(train_step_2 ,feed_dict={xs :x_data1 ,ys :y_data1})

    if i % 500 0= =0:
        print(sess.run(loss ,feed_dict={xs :x_data1 ,ys :y_data1}) ,i)
    # print(sess.run(ys,feed_dict={xs:x_data1,ys:y_data1}))

print(sess.run(prediction ,feed_dict={xs :x_data1 ,ys :y_data1}))
