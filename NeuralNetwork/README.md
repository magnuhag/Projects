A simple feed forward neural network written from scratch as a class in Python. 
The network allows for an arbitrary number of layers, and arbitrary layer sizes. 
By using `Autograd` the network can in principle use which ever loss and/ or activation function the user should wish, but as of now it is restricted to the ones supplied explicitly by the class. These restrictions might be altered/ lifted.
Currently the only optimizer available is mini-batch gradient descent. This may also be subject to change.

In this repo I have provided four (4) files. Two of these are just the plain code for the network (`NeuralNetwork` and `NNC.py`- these are identical but for the docstrings in the former) plus a run example in `network_example.py`. A third is a Jupyter Notebook `NeuralNetwork.ipynb` explaining not only the algorithms involved, but derive (some of) the mathematical expressions involved. It also involves a classification (MNIST) test use case comparing this network to Tensorflow. The results may not shock you. 
Finally I have also began working on my own automatic differentiation/ back propagation implementation to use with the neural network. This is currently being done in the `Wngert_list.ipynb` notebook, the purpose of which is to make a notebook similar to the Neural Network notebook. The final implementation will ofcourse be code in a .py file. Or I might write it in C++ for added speed - time will tell. The reason for writing this implementation is because of how slow it is do to back propagation with Autograd the way I currently am; a whole bunch of Jacobians must be evaluated, and not using the structure of the network itself to implement automatic differentiation and back prop is insane. This implementation will be in development for some time.

Since the network is currently in development, several problems are currently known:

  -Divide by zero error can occurr (very dependent upon architecture of network) when calculating gradients and/ or Jacobians. Don't know why this is. Might fix itself with my homecooked automatic diff implementation.
  
  -Bad (no) handling of exploding and vanishing gradients. Will look into gradient clipping, and other techniques.
  
  -Bad formatting and sometimes "non-Pythonic" code. I'm on it.
  
  -The network learns rather slowly. Will eventually add batch norm, dropout, and possibly more. All of this in the far future. Might also be a bug somehwere in the update of biases. Or I use too small batches for training. 
  
  -The use of python lists as "tensors". That is, the weights between two layers are represented by matrices of the type $M\in\mathbb{R}^{n\times m}$. These matrices are contained in lists. This is because it is the easiest way to do it, at least that I've found. But I don't like the solution. This is a minor issue.
  
  -The misuse of the word "tensor". A tensor is a very specific mathematical object, not just any n-dimensional array. An object $T_{j_1,\cdots,j_q}^{i_1,\cdots,i_p}$ is a tensor if the following transformation is true (for an order $p+q$ tensor with $p$ contravariant indices and $q$ covariant indices):
  
$$
\hat{T}_{j'_1,\cdots,j'_q}^{i'_1,\cdots,i'_p}=(R^{-1})\_{i_1}^{i'_1}  \cdots  (R^{-1})\_{i_p}^{i'_p}T\_{j_1,\cdots,j_q}^{i_1,\cdots,i_p}R\_{j'_1}^{j_1}\cdots R\_{j'_q}^{j_q}
$$
  
-A bunch of other minor -- and just slightly less significant than the aforementioned -- stuff that will take the backseat for quite a while longer  

-My tendancy to write sarcastic READMEs. I will *try* to work on this.
