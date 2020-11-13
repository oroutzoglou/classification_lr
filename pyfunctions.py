import numpy as np

def sigmoid(z):   # computes the sigmoid(z) of scalar/numpy array z of any size
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):  # creates a vector of zeros of shape (dim,1) and initialize b to 0
    w = np.zeros([dim,1])
    b = 0
    assert(w.shape == (dim, 1))   # triger if not true
    assert(isinstance(b, float) or isinstance(b, int))   
    return w, b
    
def propagate(w, b, X, Y):        # implement the cost function and its gradient for the propagation
                                  # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
                                  # b -- bias, a scalar
                                  # X -- data of size (num_px * num_px * 3, number of examples)
                                  # Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
                                  # returns:
                                  # cost -- negative log-likelihood cost for logistic regression
                                  # dw -- gradient of the loss with respect to w, thus same shape as w
                                  # db -- gradient of the loss with respect to b, thus same shape as b

    m = X.shape[1]    

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) +b)     # compute activation
    cost = -1/m * (np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1 - A).T))   
  
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = 1 / m *(np.sum(A - Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,     # gradients=diferential of weights and bias
             "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    # function to optimize w and b by running a gradient descent algorithm
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of shape (num_px * num_px * 3, number of examples)
    # Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    # num_iterations -- number of iterations of the optimization loop
    # learning_rate -- learning rate of the gradient descent update rule
    # print_cost -- True to print the loss every 100 steps
    # Returns:
    # params -- dictionary containing the weights w and bias b
    # grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    # costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    costs = []
    
    for i in range(num_iterations):
               
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    # predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)    
    # returns:
    # Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) +b)  
    
    for i in range(A.shape[1]):    
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0][i]>0.5):
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0
                
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction