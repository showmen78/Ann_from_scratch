


import os
from tokenize import ContStr

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image

# x_train=np.array([[1,2,3],[4,5,6],[7,8,9],[4,7,8]]).reshape(4,3)
# y_train= np.array([[0,1,1]]).reshape(1,3)
# y_pred= np.array([[.8,.3,.7]]).reshape(1,3)


#utility function
def relu(z):
    return np.maximum(z,0)

def derivative_of_relu(z):
    return np.array(z>0,dtype='float')

def sigmoid(z):
    return 1/(1+np.exp(-z))

#  implementing the model

class Layer:
    def __init__(self,w_size,last_layer):
        self.w= np.random.randn(w_size[1],w_size[0])
        self.b= np.zeros((w_size[1],1))
        self.last_layer= last_layer
        self.z=[]
        self.dz=[]
        self.dw=[]
        self.db=[]
        self.A= []


    def forward_propagation(self,A):
        self.z= np.array(self.w.dot(A)+ self.b)
        if self.last_layer:
            self.A= np.array(sigmoid(self.z))
            #print(" Y_pred : ",self.A)
        else:
            self.A= np.array(relu(self.z))
        
        return self.A

    def calculate_cost(self,m,y):
         return np.sum((y*np.log(self.A+.1))+(1-y)*np.log(1-self.A+.1))*(-1/m)

    def back_propagation(self,y,w_next,dz_next,A_prev,m):
        if self.last_layer:
            self.dz= np.array(self.A-y)
        else:
            self.dz= np.array(w_next.T.dot(dz_next)) * derivative_of_relu(self.z)
        self.dw= (1/m)*(self.dz.dot(A_prev.T))

        self.db= (1/m)*np.sum(self.dz,axis=1,keepdims=True)

        return self.w,self.dz

    def update_parameters(self,lr):
        self.w = self.w- (lr*self.dw)
        self.b= self.b- (lr*self.db)




def get_data(path,im_h,im_w):
    data={}
    #importing cat data
    for file_name in os.listdir(path+'\monkey'):
        if file_name.endswith('.jpg'):
            img= Image.open(path+'/monkey/'+str(file_name)).convert('L')
            resized_img = img.resize((im_h, im_w))
            img_arr= np.array(resized_img)
            img_arr= img_arr.reshape(im_w*im_h,1)
    
            data[str(file_name)]= {'img':img_arr,'label':1}

    #importing dog image
    for file_name in os.listdir(path+'\dog1'):
        if file_name.endswith('.jpg'):
            img= Image.open(path+'/dog1/'+str(file_name)).convert('L')
            resized_img = img.resize((im_h, im_w))
            img_arr= np.array(resized_img)
            img_arr= img_arr.reshape(im_w*im_h,1)
    
            data[str(file_name)]= {'img':img_arr,'label':0}


    #get the key list of the dict train_img
    keys_list = list(data.keys())
    #shuffle the key list
    random.shuffle(keys_list)

    x=[]
    y=[] 
    for keys in keys_list:
        x.append(np.array(data[keys]['img']))
        y.append(data[keys]['label'])
     
    x= np.array(x)
    x=np.squeeze(x)
    y= np.array(y)
    y= np.squeeze(y)
    y= y.reshape(1,x.shape[0])
 
    #print(y.shape)

    return x.T/255,y





def model(layer_dims,x_train,y_train,x_test,y_test,lr,iter):
    layers=[]
    cost =[]
    m=x_train.shape[1]

    #initiating layers
    for l in range(1,len(layer_dims)):
        layers.append(Layer([layer_dims[l-1],layer_dims[l]],l==(len(layer_dims)-1)))

    for i in range(iter):
        #forward_propagation
        A=x_train
        for l in layers:
            A=l.forward_propagation(A)
        
        #calculate cost (only last layer)
        cost.append(layers[len(layers)-1].calculate_cost(m,y_train))

        #back propagation
        w_next=[]
        dz_next=[]
        for l in reversed(range(len(layers))):
            if l ==0:
                w_next,dz_next=layers[l].back_propagation(y_train,w_next,dz_next,x_train,m)
            else:
                w_next,dz_next=layers[l].back_propagation(y_train,w_next,dz_next,layers[l-1].A,m)

        #update_parameters(lr):
        for l in layers:
            l.update_parameters(lr)

        if i%1==0:
            print('\n iter:{} \t cost:{} \t train_acc:{} \t test_acc:{}'.format(i,cost[i],
            predict(x_train,y_train,layers,m),predict(x_test,y_test,layers,x_test.shape[1])))

       
    

    return layers,cost


def predict(x,y,layers,m):
    A=x
    #running forward propagation
    for l in layers:
        A=l.forward_propagation(A)

    #checking accuracy
    y_pred= np.array(A>.5,dtype='float')
    accuracy=np.sum(y==y_pred,dtype='float')/m


    return accuracy



    

train_img_path='F:\\train'
test_img_path= 'F:\\test'
im_h=im_w=300

#getting training and testing data
x_train,y_train=get_data(train_img_path,im_h,im_w)
x_test,y_test= get_data(test_img_path,im_h,im_w)


# print("training data  x: {}    y:{}".format(x_train.shape,y_train.shape))
# print("test data  x: {}    y:{}".format(x_test.shape,y_test.shape))
    


layer_dims= [x_train.shape[0],100,50,1]
layer,cost=model(layer_dims,x_train,y_train,x_test,y_test,0.0005,100)








# predict(x_train,y_train,layer,3)




        



    


