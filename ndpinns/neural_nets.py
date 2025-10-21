"""
Filename: neural_nets.py
Author: Aoibh Schumann
Date: 2025-10-12
Version: 1.0
Description:
    feed-forward neural networks with given geometry and activation functions.
    When called with weights and biases (i.e. parameters) the neural net
    returns a function on jax.numpy arrays
"""

from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jr
import json

def generate_forward(activation):

    def forward(params):
        def u(inputs):
            # Forward Pass
            for W, b in params[:-1]:
                outputs = jnp.dot(inputs, W) + b
                inputs = activation(outputs)
            # Final inner product
            W, b = params[-1]
            outputs = jnp.dot(inputs, W) + b
            return outputs
        return u

    return forward


@dataclass
class NN:
    """
    contains information about a network

    parameter_geometry: shape of the parameter arrays
    in_dim: input dimension
    out_dim: output dimension
    depth: the number of parameter matrices
    forward: execution of the nn given parameters and point(s)
    """
    parameter_geometry : ((int, int))
    forward : callable
    in_dim : int
    out_dim : int
    depth : int

    def __init__(self, parameter_geometry, forward):
        self.parameter_geometry = parameter_geometry
        self.forward = forward
        self.in_dim = parameter_geometry[0][0]
        self.out_dim = parameter_geometry[-1][-1]
        self.depth = len(parameter_geometry)


    def __call__(self, params):
        return self.forward(params)

@dataclass
class Siren(NN):
    def __init__(self, parameter_geometry):
        super().__init__(parameter_geometry = parameter_geometry, forward = generate_forward(activation=jnp.sin))


def initialize_parameters(key, parameter_geometry, w0=1.):
    """
    initialize a neural nets' parameter values based on architecture

    we use the uniform xavier/glorot initialization (fact check)
    and we define the function iductively
    """

    # Base case
    if not parameter_geometry:
        return tuple()

    # Inductive step
    d_in, d_out = parameter_geometry[0]
    half_interval = jnp.sqrt(6/(d_in + d_out))/w0

    w = jr.uniform(key,
                   (d_in, d_out),
                   minval = -half_interval,
                   maxval = +half_interval)
    b = jr.uniform(key,
                   (d_out,),
                   minval = -half_interval,
                   maxval = +half_interval)

    parameter_geometry = parameter_geometry[1:]

    return ((w,b),) + initialize_parameters(key, parameter_geometry, w0)

def params_to_json(filename, params):
    list_params = [[param[0].tolist(), param[1].tolist()] for param in params]
    with open(filename, "w") as f:
        json.dump(list_params, f, indent=4)

def json_to_params(filename):
    with open(filename, "r") as f:
        list_params = json.load(f)
    params = tuple(
                (jnp.array(param[0]), jnp.array(param[1]))
                for param in list_params
            )
    return params
