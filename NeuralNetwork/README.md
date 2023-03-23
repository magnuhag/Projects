A simple feed forward neural network written from scratch in Python. 
The network allows for an arbitrary number of layers, and arbitrary layer sizes. 
By using Autograd the network can in principle use which ever loss and/ or activation function the user should wish, but as of now it is restricted to the ones supplied explicitly by the code. These restrictions may be altered/ lifted.
Currently the only optimizer available is mini-batch gradient descent. This may also be subject to change.

In this repo I have provided two files. One is just the plain code for the network, and the second a Jupyter Notebook explaining not only the algorithms involved but derive (sort of) the mathematical expressions involved. It also involves a classification (MNIST) test use case comparing this network to Tensorflow. The results may not shock you.

The network is currently under development, and you (the reader) may wish to keep that in mind. Several problems are currently known:

  -Divide by zero error can (very dependent upon architecture of network) occurr when calculating gradients and/ or Jacobians.
  
  -There is something wrong with the evaluation of the loss function "softmax".
  
  -An attempt will be made to implement automatic differentiation directly instead of indirectly; this is already done with Autograd, and is phenomenally inefficient. An attempt will be made. 
  
  -Bad (no) handling of exploding and vanishing gradients. 
  
  -Bad formatting and sometimes "non-Pythonic" code. I'm on it.
  
  -The use of lists as "tensors". I do not like it.
  
  -The misuse of the word "tensor". A tensor is a very specific mathematical object, not just any n-dimensional array. An object $T_{j_1,\cdots,j_q}^{i_1,\cdots,i_p}$ is a tensor if the following transformation is true (for an order $p+q$ tensor with $p$ contravariant indices and $q$ covariant indices):
  
$$
\hat{T}_{j'_1,\cdots,j'_q}^{i'_1,\cdots,i'_p}=(R^{-1})\_{i_1}^{i'_1}  \cdots  (R^{-1})\_{i_p}^{i'_p}T\_{j_1,\cdots,j_q}^{i_1,\cdots,i_p}R\_{j'_1}^{j_1}\cdots R\_{j'_q}^{j_q}
$$
  
-A bunch of other minor -- and just slightly less significant than the aforementioned -- stuff that will take the backseat for quite a while longer  

-My tendancy to write sarcastic READMEs. I will try to work on this.
