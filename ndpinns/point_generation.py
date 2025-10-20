"""
Filename: point_generation.py
Author: Aoibh Schumann
Date: 2025-10-12
Version: 1.0
Description:
    Generates points for training given a domain and desired refinement. Also
    packages together DomainConditions and points for training into the
    dataclass TrainingPoints.
"""

from dataclasses import dataclass
import jax.numpy as jnp
import jax.random as jr
import itertools as it

import domains
import conditions



def _zero_dimension_special_case(domain):
    """
    handles the special case where we have a point condition. generates a
    single point
    """
    points = jnp.array((domain.embedding(),))

    return points


def generate_grid_points(domain: "Domain", refinement: int):
    """
    generate equi-spaced training points within a given domain
    """

    if domain.dimension == 0:
        return _zero_dimension_special_case(domain)

    bounds = domain.bounds
    embedding = domain.embedding

    points_by_dim = (jnp.linspace(bound[0],
                                  bound[1],
                                  refinement) for bound in bounds)
    pre_points = tuple(it.product(*points_by_dim))
    pre_points = (jnp.stack(point) for point in pre_points)

    points = (embedding(point) for point in pre_points)
    points = jnp.stack(tuple(points))

    return points


def generate_random_points(domain: "Domain", refinement: int, key):
    """
    generate random training points within a given domain
    """
    if domain.dimension == 0:
        return _zero_dimension_special_case(domain)


    bounds = domain.bounds
    embedding = domain.embedding
    dimension = domain.dimension

    n = refinement**dimension
    minval = jnp.array((bound[0] for bound in bounds))
    maxval = jnp.array((bound[1] for bound in bounds))
                                                           
    pre_points = jr.uniform(key=key,
                            shape=(n, dimension),
                            minval=minval,
                            maxval=maxval)
                                                           
    points = (embedding(point) for point in pre_points)
                                                           
    return points

@dataclass
class TrainingPoints:
    domain_condition : "DomainCondition"
    points : "ArrayLike"


def generate_training_points(domain_condition: "DomainCondition",
                             refinement: int,
                             generator = generate_grid_points
                             ) -> "TrainingPoints":
    """
    generate training points for a given domain condition and return the
    results as a member of the TrainingPoints class
    """
    domain = domain_condition.domain
    points = generator(domain=domain, refinement=refinement)
    
    training_points = TrainingPoints(domain_condition=domain_condition, points=points)

    return training_points

def generate_problem_points(problem: "DomainConditionProblem",
                            refinement: int | (int)):
    """
    generate training points in a grid for the domains in the given problem
    """
    domain_conditions = problem.domain_conditions
    if isinstance(refinement, int):
        refinement = len(domain_conditions)*(refinement,)
    elif isinstance(refinement, tuple):
        assert len(refinement) == len(domain_conditions)
        assert all((isinstance(_, int) for _ in refinement))
    else:
        raise TypeError("refinement must be an int or a tuple of ints")


    args = zip(domain_conditions, refinement)


    output = tuple(generate_training_points(*arg) for arg in args) 
    return output

