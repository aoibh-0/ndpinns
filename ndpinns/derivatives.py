"""
Filename: derivatives.py
Author: Aoibh Schumann
Date: 2025-10
Version: 1.0
Description:
    Defines derivative operators on functions, particularly neural networks
    defined with JAX. Each operator takes a parameter "u" that is assumed to
    act on jax.numpy arrays. Each operator here returns another function of
    jax.numpy operators.
"""

from jax import jvp
import jax.numpy as jnp

def d(u, direction):
    """
    returns the derivative of fun in the given direction
    """
    def du(x):
        # create units
        tangents = jnp.zeros_like(x)
        tangents = tangents.at[...,direction].set(1.)

        # define derivative
        du_function = jvp(fun=u, primals=(x,), tangents=(tangents,))[1]

        return du_function

    return du



def laplacian(u):
    """
    returns the laplacian of a function u (assuming euclidean coordinates)
    """
    def nabla_u(x):
        x = jnp.array(x)
        second_derivatives = (d(d(u, i), i)(x) for i in range(x.shape[-1]))
        return sum(second_derivatives, jnp.zeros_like(x))
    return nabla_u



def generate_helmholtz_operator(k=1, f = None):
    """
    generates the helmholtz operator with given constant and source function
    """
    if f == None:
        f = lambda x: jnp.array((0,))
    def helmholtz_operator(u):
        nabla_u = laplacian(u)
        helmholtz_u = lambda x: nabla_u(x) + k*u(x) + f(x)
        return helmholtz_u
    return helmholtz_operator


