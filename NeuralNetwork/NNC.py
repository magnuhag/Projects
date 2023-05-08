import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad as egrad

class NeuralNet:
    
    def __init__(self):
        self.layers = []
        self.actFuncs = []
        self.weights = []
        self.biases = []
        self.Z = []
        self.A = []
        self.delta = []

    def add(self, nNeurons, actFunc, inputSize = None):

        if isinstance(nNeurons, int) and nNeurons >= 1:
            self.layers.append(nNeurons)

        else:
            #Should be obvious to anyone attempting to use this class. 
            #Still, might catch a typo
            raise TypeError("nNeurons must be of type int and greater\
                             than or equal to 1")

        if isinstance(inputSize, int):
            #Haven't really discussed initialization
            self.weights.append(np.random.randn(inputSize, nNeurons)*0.01)

        elif isinstance(inputSize, type(None)):
            self.weights.append(np.random.randn(self.layers[-2], nNeurons)*0.01)
        #Errrrr, I'll get back to this
        else:
            raise TypeError("Errr")

        if isinstance(actFunc, str):
            function = self.activation_function(actFunc)
            self.actFuncs.append(function)
        else:
            raise TypeError("actFunc argument must be of type str")

        self.biases.append(np.random.randn(nNeurons)*0.01)


    def activation_function(self, act):

        if act == "sigmoid":
            activ = lambda x: 1/(1+np.exp(-x))

        elif act == "RELU":
            activ = lambda x: np.maximum(x, 0)

        elif act == "leaky_RELU":
            activ = lambda x: np.maximum(x, 0.01 * x)

        elif act == "softmax":
            activ = lambda x: np.exp(x)/np.sum(np.exp(x))

        elif act == "linear":
            activ = lambda x: x
        
        else:
            raise NameError("No activation function named %s" %act)

        return activ

    def loss_function(self, loss):

        if isinstance(loss, str):
            if loss == "MSE":
                func = lambda x, y: np.mean((x - y)**2, axis = 0)
            elif loss == "categorical_cross":
                func = lambda x,y: -np.sum(y*np.log(x), axis = 0)
            else:
                raise ValueError("Invalid loss function name")
        else:
            raise TypeError("Loss function argument must be of type str")

        return func

    def feed_forward(self, X):

        #Feeding in feature matrix
        self.Z[0] = X @ self.weights[0] + self.biases[0].T
        #Activation in first hidden layer
        self.A[0] = self.actFuncs[0](self.Z[0])

        for i in range(1, len(self.weights)):
            #Feeding forward
            self.Z[i] = self.A[i-1] @ self.weights[i] + self.biases[i].T
            self.A[i] = self.actFuncs[i](self.Z[i])

    def diff(self, C, A):

        dCda = egrad(C)
        dAdz = jacobian(A)

        return dCda, dAdz

    def back_prop(self, y, diff):

        #Assigning Jacobian and gradient functions as variables
        dC, da = diff
        #"Empty" (Zeros) array to hold Jacobian
        dAct = np.zeros(len(self.Z[-1]))
        #Empty array to hold derivative of cost function
        dcda = dAct.copy()
        #Empty array to hold delta^L
        self.delta[-1] = np.zeros((len(self.Z[-1]), self.layers[-1]))
        for i in range(len(self.Z[-1])):
            #Calculate Jacobian and derivative for each training example in batch
            dAct = da(self.Z[-1][i])
            dcda = dC(self.A[-1][i], y[i])
            #Jacobian of activation times derivative of cost function
            self.delta[-1][i] = dAct @ dcda

        for i in range(len(self.weights)-2, -1, -1):
            #Gradient of activation function of hidden layer i
            dfdz = egrad(self.actFuncs[i])
            #Equation 2 is calculated in 2 parts. Just for ease of reading
            t1 =  self.delta[i+1] @ self.weights[i+1].T
            self.delta[i] = np.multiply(t1, dfdz(self.Z[i]))

    def optimizer(self, X, eta):

        self.weights[0] -= eta * (X.T @ self.delta[0])
        self.biases[0] -= eta * np.sum(self.delta[0], axis = 0)

        for i in range(1, len(self.weights)):
            self.weights[i] -= eta * (self.A[i-1].T @ self.delta[i])
            self.biases[i] -= eta * np.sum(self.delta[i], axis = 0)

    def train(self, X, y, epochs, loss, metric, batchSize = 10,
              numIters = 100, etaInit = 10**(-4), decay = 0.1):
        #Creating lists to contain A, Z, and Delta matrices.
        self.A = self.Z = self.delta = [0]*len(self.layers)
        
        diff = self.diff(self.loss_function(loss), self.actFuncs[-1])

        dataIndices = len(X)
        #eta function: for decreasing learning rate as training progresses
        eta = lambda etaInit, iteration, decay: etaInit/(1+decay*iteration)

        for i in range(1, epochs+1):
            for j in range(numIters):
                eta1 = eta(etaInit, j, decay)
                #Randomly choose datapoints to use as mini-batches
                chosenDatapoints = np.random.choice(dataIndices, 
                                                     size = batchSize, 
                                                     replace = False)
                #Making mini-batches
                XMini = X[chosenDatapoints]
                yMini = y[chosenDatapoints]
                #Feed forward
                self.feed_forward(XMini)
                #Backprop
                self.back_prop(yMini, diff)
                #Update weights and biases
                self.optimizer(XMini, eta(etaInit, j, decay))

            #Make a prediction and print mean of performance and loss of mini-batch
            predicted = self.predict(XMini)
            metricVal = np.mean(self.metrics(predicted, yMini, metric))
            lossVal = np.mean(self.loss_function(loss)(predicted, yMini))
            print("mean loss = %.3f ---------- %s = %.2f at epoch %g" 
                  %(lossVal, metric, metricVal, i))

    def metrics(self, yHat, y, a):

        if a == "accuracy":
            s = 0
            for i in range(len(y)):
                true = np.argmax(y[i])
                pred = np.argmax(yHat[i])
                if true == pred:
                    s += 1
                else:
                    continue

            return s/len(yHat)

        elif a == "MSE":
            return np.mean((y-yHat)**2, axis = 0)


    def predict(self, X):
        
        self.feed_forward(X)
        return self.A[-1]