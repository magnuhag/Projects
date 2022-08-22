A simple feed forward neural network written from scratch in Python. 
The network allows for an arbitrary number of layers, and arbitrary layer sizes. 
By using Autograd the network can in principle use which ever loss and/ or activation funcion the user should wish, but as of now it is restricted to the ones supplied explicitly by the code. These restrictions may be altered/ lifted.
Currently the only optimizer available is mini-batch gradient descent. This may also be subject to change.

In this repo I have provided two files. One is just the plain code for the network, and the second a Jupyter Notebook explaining not only the algorithms involved but derive (sort of) the mathematical expressions involved. It also involves a classification (MNIST) test usecase comparing this network to Tensorflow. The results may not shock you.

The network is currently under development and you (the reader) may wish to keep that in mind. Several problems are currently known:
  -Divide by zero error when calculation some gradients and/ or Jacobians.
  -Bad (no) handling of exploding and vanishing gradients.
  -Bad formating and sometimes "non-Pythonic" code. 
  -The use of lists as tensors. I do not like it.
  -The misuse of the word "tensor". This is a (rank 2) tensor
  $$
  A^{ij}=\frac{\partial x_i'}{\partial x_k}\frac{\partial x_j'}{\partial x_l}A^{kl}
  $$
  