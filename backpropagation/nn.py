from neural_networks.backpropagation.network import Value
import random

class Neuron:
  def __init__(self, nin):
    """
    Initialization of the Neuron object

    :nin (int): number of input values
    """
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    """
    Forward pass of the Neuron object
    """
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    """
    Determine the parameters of the Neuron object (the weights)
    """
    return self.w + [self.b]

class Layer:
  def __init__(self, nin, nout):
    """
    Initialization of the Layer object (collection of Neurons)

    :nin (int): number of input values (number of weights in Neuron)
    :nout (list): list of number of output values (number of Neurons in Layer)
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    """
    Call function n(x) on the layer obect

    :x (list): list of input values
    """
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    """
    Determine the parameters of the Layer object
    """
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  def __init__(self, nin, nouts):
    """
    Initialize the MLP object, a collection of Layers
    
    :nin (int): number of input values
    :nout (list): list of number of output values (number of Neurons in each individual layer)
    """
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    """
    Call function n(x) on the MLP object 

    :x (list): list of input values
    """
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    """
    Determine the parameters of the MLP object
    """
    return [p for layer in self.layers for p in layer.parameters()]