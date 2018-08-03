import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr,formataddr
from email import encoders
import smtplib

x_test=np.loadtxt('./obs_normal_test.txt')
y_test=np.loadtxt('./parameters_for_test.txt')

x_data0 = np.loadtxt("./obs_normal_train.txt")
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