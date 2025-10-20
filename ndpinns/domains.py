"""
Filename: domains.py
Author: Aoibh Schumann
Date: 2025-10
Version: 1.0
Description:
    domains are k-cells embedded in a (possibly higher) dimensional space. The
    embedding mapping gives more interesting shapes than would otherwise be
    possible with a rectangular subset of euclidean space. N.B. the chosen
    embedding will result in point densities that are not constant (assuming a
    uniform distribution of points in the k-cell).
"""


from collections.abc import Callable
from dataclasses import dataclass
import jax.numpy as jnp



@dataclass
class kCell:
    dimension: int
    bounds: ((float, float))

@dataclass
class Domain(kCell):
    embedding : Callable[[(float)], (float)] = lambda x: x # embedding of the kCell into another space



def dimensional_boundaries(domain : "Domain", d : int) -> ("Domain"):
    """
    find the boundaries of a domain along a given axis d

    primarily a helper function for the full boundaries function
    """
    dimension = domain.dimension
    bounds = domain.bounds
    embedding = domain.embedding

    surface_bounds = tuple(bounds[:d] + bounds[d+1:])
    surface_cell = kCell(dimension = dimension-1,
                         bounds = surface_bounds)
    surface_embeddings = (
            lambda x: jnp.insert(x, d, bounds[d][0], axis = -1),
            lambda x: jnp.insert(x, d, bounds[d][1], axis = -1)
                        )

    surface_domains = tuple(Domain(dimension=dimension-1,
                                   bounds=surface_bounds,
                                   embedding=surface_embedding
                              )
                       for surface_embedding in surface_embeddings)
    return surface_domains



def boundaries(domain : "Domain") -> ("Domain"):
    """
    return a list of domains that bound the given domain
    """
    dimension = domain.dimension
    boundary_domain_pairs = (
            tuple(dimensional_boundaries(domain=domain, d=d))
            for d in range(dimension)
                            )
    boundary_domains = sum(boundary_domain_pairs, ())
    return boundary_domains


