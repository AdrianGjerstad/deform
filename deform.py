#!/usr/bin/env python3

import random as r
import math

class DeformNeuralNetwork:
  def __init__(self, node_cts=[2, 2, 1]):
    for i in range(len(node_cts)):
      if not isinstance(node_cts[i], int):
        raise TypeError('Innapropriate type for DeformNeuralNetwork \'node_cts[%i]\' argument: %s (expected int)' % (i, type(inputs).__name__))

    self.nodes = []

    if len(node_cts) < 2:
      raise TypeError('Not enough layers for neural network to function. (Minimum of 2)')

    for i in range(len(node_cts)):
      self.nodes.append([])
      for j in range(node_cts[i]):
        if i == 0:
          self.nodes[i].append(_NNN())
        else:
          self.nodes[i].append(_NNN(len(self.nodes[i-1])))

  def __repr__(self):
    result = 'DeformNeuralNetwork at <%s>: {\n' % (hex(id(self)))

    for i in range(len(self.nodes)):
      if i == 0:
        result += '  Inputs: {\n'
        for j in range(len(self.nodes[i])):
          result += '    Node[%i]: %s\n' % (j, repr(self.nodes[i][j]))
        result += '  },\n'
      elif i == len(self.nodes) - 1:
        result += '  Outputs: {\n'
        for j in range(len(self.nodes[i])):
          result += '    Node[%i]: %s\n' % (j, repr(self.nodes[i][j]))
        result += '  }\n'
      else:
        result += '  HiddenLayer[%i]: {\n' % (i - 1)
        for j in range(len(self.nodes[i])):
          result += '    Node[%i]: %s\n' % (j, repr(self.nodes[i][j]))
        result += '  },\n'

    result += '}'

    return result

  def ff(self, inputs):
    for i in range(len(inputs)):
      self.nodes[0][i].value = inputs[i]

    for i in range(len(self.nodes)):
      for j in range(len(self.nodes[i])):
        if i != 0:
          self.nodes[i][j].run(self.nodes[i-1])

    result = []
    for i in range(len(self.nodes[-1])):
      result.append(self.nodes[-1][i].value)

    return result

  def fit(self, training):
    error = 0
    output = 0
    input_ = 0
    for unit in training:
      actual = self.ff(unit[0])

      output += sum(actual)/len(actual)
      input_ += sum(unit[0])/len(unit[0])

      for i in range(len(unit[1])):
        error += unit[1][i] - actual[i]

    change_by = sigmoid(error)*sigmoid(input_)*sigmoid(sigmoid_deriv(output))/2

    for i in range(len(self.nodes)):
      for j in range(len(self.nodes[i])):
        if i != 0:
          for k in range(len(self.nodes[i][j].weights)):
            self.nodes[i][j].weights[k] += change_by

class _NNN:
  def __init__(self, inputs=0):
    self.weights = []

    for i in range(inputs):
      self.weights.append(r.random())

    self.value = None

  def run(self, inputs=[]):
    sum_result = 0
    for i in range(len(self.weights)):
      sum_result += inputs[i].value*self.weights[i]

    result = sigmoid(sum_result)

    self.value = result

  def __repr__(self):
    return '(weights=%s, value=%s)' % (str(self.weights), str(self.value))

def sigmoid(x):
  return 1/(1+(math.e**-x))

def sigmoid_deriv(x):
  return x * (1 - x)

def neural(node_cts=[2, 2, 1]):
  return DeformNeuralNetwork(node_cts)
