# neural_networks
A neural network project based on GPT-2

1) First, an investigation into the ways how neural networks work, through the eyes of backpropagation and gradient descent. This can be found in the 'backpropagation' folder, where we implement a "bare-bones" version of pytorch, using Values instead of Tensors and a backwards backpropagation algorithm to calculate the gradients of all of the neurons. Then, a further investigation into manual backpropagation to advance our knowledge on the most important algorithm in machine learning, and specifically, neural networks. 

2) Secondly, the creation of a generation model called makemore. As the name suggests, Makemore is about making more from data that was given. In this case, we take a dataset of names and are trying to generate new names from the data in multiple different ways. These include: 
    * A Bigram Model
    * A MLP ![A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (using trigrams with tanh activation functions to create outputs with multinomial choosing)
    * Wavenet Structure ![WaveNet: A Generative model for Raw Audio](https://arxiv.org/pdf/1609.03499) (trying to incorporate the model of using multiple hidden layers in order to tune hyperparameters and increase the complexity)
    * A RNN, a GRU, and finally, the current most popular way (circa 2024), a Transformer. Through this, we will train a neural network to create brand new, readable names that are understandable and structured.

3) Finally, we'll create a version of GPT. Using a LLM, we will build a GPT tokenizer in order to help generate data for the model, as well as use a complicated system of neural networks in order to build an algorithm that will allow us to create ChatGPT right from our own browser. 

![neural_network](https://github.com/kev374k/neural_networks/assets/54005848/366c0ef7-a2df-4bbd-853e-383faaaa5939)
