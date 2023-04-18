import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad as egrad

class NeuralNet:
    """
    A class to create a feed forward neural network.
    All layers are fully connected to adjacent layers.
    In a layer l, all neurons share the same activation
    function. Different layers can have different
    activation functions. Layers are added sequatially
    with the add method.
    Backpropagation is done with the autograd module,
    where jacobians and gradients are calculated using
    automatic differentiation.


    Attributes
    ----------
    None

    Methods
    -------
    add(nNeurons, actFunc, inputSize = None):
        Adds a dense layer of neurons fully connected to the
        previous layer. User can spesify number of neurons
        and activation function.

    activation_function(act):
        Contains activation functions. No need
        for user to access this method.

    loss_function(loss):
        Contains loss functions. No need for user
        to access this method.

    feed_foward(X):
        Feeds data through the network.
        No need for user to access this method.

    diff(C, A):
        Calculates the gradient of C and
        jacobian of A. No need for user
        to access this method.

    backprop(y, diff):
        Performs backprop. No need for user
        to access this method.

    optimizer(X, eta):
        Mini-batch gradient descent. No need for
        user to access this method.

    train(X, y, epochs, loss, metric, batchSize = 10,
          numIters = 100, etaInit = 10**(-4), decay = 0.1):
        Trains the neural network.

    metrics(yHat, y, a):
        Calculates matrics.

    predict(X):
        Does one feed forward pass and returns
        the output.

    """
    def __init__(self):
        """__init__()
        Creates the following instance attributes
        ----------
        layers : list
            List to hold size of each layer
        actFuncs : list
            List to hold actavtion functions
            of each layer
        weights : list
            List to hold weight arrays of each layer
        biases : list
            List to hold bias arrays of each layer
        Z : list
            List to hold input of each layer
        A : list
            List to hold activation of each layer
        delta : list
            list to hold delta for each layer

        Parameters
        -------
        None

        Returns
        -------
        None
        """

        self.layers = []
        self.actFuncs = []
        self.weights = []
        self.biases = []
        self.Z = []
        self.A = []
        self.delta = []

    def add(self, nNeurons, actFunc, inputSize = None):
        """add(nNeurons, actFunc, inputSize = None)
        Sequentially adds layer to network. If first layer
        inputSize must be supplied; this is not inferred.
        Input size can be number of features in feature matrix, or
        more generally the length of axis 1 of X.
        That is, if X is

        feature 1,   feature 2, ... , feature n

        value 1,1    value 1,2        value 1,n
        value 2,1    value 2,2        value 2,n
        .            .                .
        .            .                .
        .            .                .
        value m,1    value m,2        value m, n

        the inputSize value is equal to n.


        Parameters
        -------
        nNeurons : int
            number of neurons in layer
        actFunc : str
            activation function to be used in
            layer. See activation_function method
            for available functions.
        inputSize : int
            If not first layer, must be None.

        Returns
        -------
        None
        """

        if isinstance(nNeurons, int) and nNeurons >= 1:
            self.layers.append(nNeurons)

        else:
            #Should be obvious to anyone attempting to use this class. Still: might catch a typo
            raise TypeError("nNeurons must be of type int and greater than or equal to 1")

        if isinstance(inputSize, int):
            #I haven't really discussed initialization of weights and biases. Upsies
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

        #For each added layer we append 0 to the lists
        #so that they get the appropriate length
        self.A.append(0)
        self.Z.append(0)
        self.delta.append(0)

    def activation_function(self, act):
        """activation_function(act)
        Available activation functions for use in
        layers.
        Available activation functions:

        -sigmoid
        -RELU
        -leaky_RELU
        -softmax
        -linear (y=x)

        !!softmax not supported as activation function in
        anything but output layer!!

        Parameters
        -------
        nNeurons : int
            number of neurons in layer
        act : str
            name of activation function to be used in layer

        Returns
        -------
        activ : function
                the function specified by act
                parameter
        """

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
        """loss_function(loss)
        Available loss functions for evaluating
        error.
        Available activation functions:

        -MSE (mean squared error)
        -categorical_cross (-entropy)

        Parameters
        -------
        loss : str
            loss function to be used

        Returns
        -------
        func : function
                the function specified by loss
                parameter
        """

        if isinstance(loss, str):
            if loss == "MSE":
                func = lambda x, y: np.mean((x - y)**2, axis = 0, keepdims = True)
            elif loss == "categorical_cross":
                func = lambda x,y: -np.sum(y*np.log(x), axis = 0)
            else:
                raise ValueError("Invalid loss function name")
        else:
            raise TypeError("Loss function argument must be of type str")

        return func

    def feed_forward(self, X):
        """feed_forward(X)
        Does feed forward pass of input data

        Parameters
        -------
        X : numpy array
            Data to be passed though network

        Returns
        -------
        None
        """
        #Feeding in feature matrix
        self.Z[0] = X @ self.weights[0] + self.biases[0].T
        #Activation in first hidden layer
        self.A[0] = self.actFuncs[0](self.Z[0])

        for i in range(1, len(self.weights)):
            #Feeding forward
            self.Z[i] = self.A[i-1] @ self.weights[i] + self.biases[i].T
            self.A[i] = self.actFuncs[i](self.Z[i])

    def diff(self, C, A):
        """diff(C, A)
        Calculates gradient and jacobian of
        C and A (cost function, and activation in
        output layer)

        Parameters
        -------
        C : function
            cost/ loss function
        A : function
            activation function in last layer

        Returns
        -------
        dCda : function
            gradient of C
        dAdz : function
            jacobian of A
        """
        dCda = egrad(C)
        dAdz = jacobian(A)

        return dCda, dAdz

    def back_prop(self, y, diff):
        """back_prop(y, diff)
        Performs back propagation using
        gradients and jacobians found by autograd.

        Parameters
        -------
        y : numpy array
            targets.
        diff : tuple
            holds the functions dC,da, calculated
            in the diff method.

        Returns
        -------
        None
        """
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
            #Gradient of activation function of hidden layer i. No need for Jacobian here
            dfdz = egrad(self.actFuncs[i])
            #Equation 2 is calculated in 2 parts. Just for ease of reading
            t1 =  self.delta[i+1] @ self.weights[i+1].T
            self.delta[i] = np.multiply(t1, dfdz(self.Z[i]))

    def optimizer(self, X, eta):
        """optimizer(X, eta)
        Updates weights and biases based on error made by
        network. Currently only supports gradient descent.

        Parameters
        -------
        X : numpy array
            input data
        eta : float
            learning rate

        Returns
        -------
        None
        """

        self.weights[0] -= eta * (X.T @ self.delta[0])
        self.biases[0] -= eta * np.sum(self.delta[0], axis = 0)

        for i in range(1, len(self.weights)):
            self.weights[i] -= eta * (self.A[i-1].T @ self.delta[i])
            self.biases[i] -= eta * np.sum(self.delta[i], axis = 0)

    def train(self, X, y, epochs, loss, metric, batchSize = 10,
              numIters = 100, etaInit = 10**(-4), decay = 0.1):
        """train(X, y, epochs, loss, metric, batchSize = 10,
                 numIters = 100, etaInit = 10**(-4), decay = 0.1)

        Trains the network. By doing feed forward, then backporp, repeat.


        Parameters
        -------
        X : numpy array
            data the network is supposed to learn
        y : numpy array
            targets
        epochs : float
            number of training epochs
        loss : str
            loss function to be used. See loss_function
            method for available ones
        metric : str
            Which metric to use. See metric method for
            available ones.
        batchSize : int
            Batch size for forward passes
        numIters : int
            number of iterations per epoch
        etaInit : float
            initial learning rate
        decay : float
            how fast learning rate decreases.

        Returns
        -------
        None
        """

        diff = self.diff(self.loss_function(loss), self.actFuncs[-1])

        dataIndices = len(X)
        #eta function (not the Dirichlet one): for decreasing learning rate as training progresses
        eta = lambda etaInit, iteration, decay: etaInit/(1+decay*iteration)

        for i in range(1, epochs+1):
            for j in range(numIters):
                eta1 = eta(etaInit, j, decay)
                #Randomly choose datapoints to use as mini-batches
                chosen_datapoints = np.random.choice(dataIndices, size = batchSize, replace = False)
                #Making mini-batches
                XMini = X[chosen_datapoints]
                yMini = y[chosen_datapoints]
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
            print("mean loss = %.3f ---------- %s = %.2f at epoch %g" %(lossVal, metric, metricVal, i))

    def metrics(self, yHat, y, a):
        """metrics(yHat, y, a)
        Calculates metric. Available ones:
        -accuracy
        -MSE (mean squared error)

        Parameters
        -------
        yHat : numpy array
            output of network
        y : numpy array
            targets
        a : str
            name of metric

        Returns
        -------
        metric : numpy array
        """

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
        """predict(X)

        Does a feed forward pass of X

        Parameters
        -------
        X : numpy array
            data to be passesd

        Returns
        -------
        A : numpy array
            activation in last layer
        """
        self.feed_forward(X)
        return self.A[-1]
