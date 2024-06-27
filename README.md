# neural_networks
A neural network project based on GPT-2

1) First, an investigation into the ways how neural networks work, through the eyes of backpropagation and gradient descent. This can be found in the 'backpropagation' folder, where we implement a "bare-bones" version of pytorch, using Values instead of Tensors and a backwards backpropagation algorithm to calculate the gradients of all of the neurons.

2) Secondly, the creation of a generation model called makemore. As the name suggests, Makemore is about making more from data that was given. In this case, we take a dataset of names and are trying to generate new names from the data in multiple different ways. These include: A Bigram Model, a Bag of Words Model, a MLP, a RNN, a GRU, and finally, the current most popular way (circa 2024), a Transformer. 
