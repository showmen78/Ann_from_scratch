#importing libraries
# import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image
# from PIL import Image

# X_train = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/train_X.csv', delimiter = ',').T
# Y_train = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/train_label.csv', delimiter = ',').T

# X_test = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/test_X.csv', delimiter = ',').T
# Y_test = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/test_label.csv', delimiter = ',').T

X_train = np.loadtxt('m_nist_dataset_for_multiclassification/train_X.csv', delimiter = ',').T
Y_train = np.loadtxt('m_nist_dataset_for_multiclassification/train_label.csv', delimiter = ',').T

X_test = np.loadtxt('m_nist_dataset_for_multiclassification/test_X.csv', delimiter = ',').T
Y_test = np.loadtxt('m_nist_dataset_for_multiclassification/test_label.csv', delimiter = ',').T

# index = random.randrange(0, X_train.shape[1])
# plt.imshow(X_train[:, index].reshape(28, 28), cmap = 'gray')
# plt.show()
data={}
for i in range(X_train.shape[1]):
    data[str(i)]= {'x':X_train[:,i],'y':Y_train[:,i]}
    
#randomizing the training data 
keys= list(data.keys())
np.random.shuffle(keys)
X=[]
Y=[]
for i in keys:
    X.append(data[i]['x'])
    Y.append(data[i]['y'])
    
X= np.array(X).T/255 #final train data #dividing by 255 to normalize the value between 0 and 1
Y= np.array(Y).T #final test data   

X_test= np.array(X_test)/255   #dividing by 255 to normalize the value between 0 and 1
Y_test= np.array(Y_test)


#utility function
def relu(z):
    return np.maximum(z,0)

def derivative_of_relu(z):
    return np.array(z>0,dtype='float')

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_of_sigmoid(z):
    sg= sigmoid(z)
    return sg*(1-sg)

def tanh(z):
    return np.tanh(z)

def derivative_of_tanh(z):
    return 1-np.power(tanh(z),2)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=0)


# The layer class
class Layer:
    def __init__(self,w_size,last_layer):
        self.w= np.random.randn(w_size[1],w_size[0])-.5
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
            #self.A= np.array(sigmoid(self.z)) #for binary classification problem
            self.A= np.array(softmax(self.z)) #for multiclassification problem
            #print(" Y_pred : ",self.A)
        else:
            self.A= np.array(sigmoid(self.z))
        
        return self.A

    def calculate_cost(self,m,y):
        
         #return np.sum((y*np.log(self.A+.1))+(1-y)*np.log(1-self.A+.1))*(-1/m) #for binary classification
        return np.sum(y*np.log(self.A))/(-1/m) #for multiclassification problem

    def back_propagation(self,y,w_next,dz_next,A_prev,m):
        if self.last_layer:
            self.dz= np.array(self.A-y)
        else:
            self.dz= np.array(w_next.T.dot(dz_next)) * derivative_of_sigmoid(self.z)

        
        self.dw= (1/m)*np.array(self.dz.dot(A_prev.T))
        self.db= (1/m)*np.sum(self.dz,axis=1,keepdims=True)

        return np.array(self.w),np.array(self.dz)

    def update_parameters(self,lr):
        w0= self.w
        self.w = self.w- (lr*self.dw)
        self.b= self.b- (lr*self.db)
        


# The main function 

def model(layer_dims,x_train,y_train,x_test,y_test,lr,iteration):
    layers=[]
    cost =[]
    m=x_train.shape[1]

    #initiating layers
    for l in range(1,len(layer_dims)):
        layers.append(Layer([layer_dims[l-1],layer_dims[l]],l==(len(layer_dims)-1)))


    for i in range(iteration):
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
            if l==0:
                w_next,dz_next=layers[l].back_propagation(y_train,w_next,dz_next,x_train,m)
            else:
                w_next,dz_next=layers[l].back_propagation(y_train,w_next,dz_next,layers[l-1].A,m)

        #update_parameters(lr):

        for l in layers:
            l.update_parameters(lr)

        if i%2==0:
            print('\n iter:{} \t cost:{} \t train_acc:{} \t test_acc:{}'.format(i,cost[i],
            predict(x_train,y_train,layers,m),predict(x_test,y_test,layers,x_test.shape[1])))

       
    

    return layers,cost

# predict 
def predict(x,y,layers,m):
    A=x
    #running forward propagation
    for l in layers:
        A= l.forward_propagation(A)
    #checking accuracy
    y_acc= np.argmax(y,0)
    y_pred= np.argmax(A,0)
    
    return np.mean(y_acc== y_pred)*100
    

#layer dimentions and learing rate
layer_dims= [X.shape[0],Y.shape[0]]
lr = .2

#calling the model (strating training)
layer,cost=model(layer_dims,X,Y,X_test,Y_test,lr,3000)



# index = random.randrange(0, X_test.shape[1])
# plt.imshow(X_test[:, index].reshape(28, 28), cmap = 'gray')

# test= np.array(X_test[:,index].reshape(X_test.shape[0],1))

# for l in layer:
#     test=l.forward_propagation(test)

# print('Actual answer is = {} and the model answer is = {}'.format(np.argmax(Y_test[:,index],0),np.argmax(test)))

# plt.show()


