"""
Filename: conditions.py
Author: Aoibh Schumann
Date: 2025-10-12
Version: 1.0
Description:
    Defines conditions for functions on domains. A condition is an operator
    that we want set to zero at all points. A condition paired with a domain is
    a DomainCondition, and a collection of DommainConditions is a Problem.
"""
from dataclasses import dataclass
import jax.numpy as jnp

from derivatives import *
from domains import *

### DomainConditions
@dataclass
class DomainCondition:
    """
    domain and a condition for a neural net on that domain

    fundamentally the same for both the interior of a domain and its
    boundary; we just change which domain we are talking about

    condition: NN -> (float) -> (float)
    """
    domain : "Domain"
    condition : "Callable[[u], Callable[[ArrayLike], ArrayLike]]"

    def loss(self, u, points):
        condition_results = self.condition(u)(points)
        return jnp.array(jnp.mean(condition_results**2))

@dataclass
class Dirichlet(DomainCondition):
    """
    implements the dirichlet condition on a boundary
    """
    domain : "Domain"
    function : "Callable[(float), (float)]"
    condition : "Callable[[callable], Callable[[ArrayLike], ArrayLike]]"

    def __init__(self, domain, function):
        self.function = function
        condition = lambda u: lambda x: jnp.array(u(x)) - jnp.array(function(x))
        super().__init__(domain = domain, condition = condition)

@dataclass
class ConstantBoundary(Dirichlet):
    """
        constant boundary condition
    """
    domain : "Domain"
    constant : (float)
    function : "Callable[[ArrayLike], ArrayLike]"
    # inherited values

    def __init__(self, domain, constant):
        self.constant = constant
        function = lambda x: self.constant
        super().__init__(domain = domain, function = function)


@dataclass
class PointSource(ConstantBoundary):
    """
    defines a point source condition

    location: a point in the domain
    constant: the nn's value there
    """
    location: (float)
    constant: (float)
    domain : "Domain"
    # inherited values
    def __init__(self, location, constant):
        self.location = location
        domain = Domain(dimension=0,
                         bounds=(),
                         embedding = lambda : location)
        super().__init__(domain=domain, constant=constant)



def dirichlet_boundaries(
        domain : "Domain",
        function : "Callable[(float), (float)]"
        ) -> ("DomainCondition"):
    """
    defines dirichlet conditions for boundaries of a given domain
    """
    bounds = boundaries(domain)
    conditions = (Dirichlet(boundary, function) for boundary in bounds)
    return conditions

def constant_boundaries(
        domain : "Domain",
        constant : (float)
        ) -> ("DomainCondition"):
    """
    constant boundary condition on boundaries of a given domain
    """
    bounds = boundaries(domain)
    conditions = (ConstantBoundary(boundary, constant) for boundary in bounds)
    return tuple(conditions)

### DomainConditionProblems
@dataclass
class DomainConditionProblem:
    name : str
    domain_conditions : ("DomainCondition")
    loss : callable

    def __init__(self, name, domain_conditions):
        self.name = name
        self.domain_conditions = domain_conditions
        def loss_fun(u, training_points):
            losses = tuple((tp.domain_condition.loss(u, tp.points)**2
                      for tp in training_points))
            loss_sum = sum(losses, jnp.array(0.,))
            return loss_sum
        self.loss = loss_fun


@dataclass
class DirichletProblem(DomainConditionProblem):
    name : str
    equation : str
    domain : "Domain"
    operator : Callable[[callable, (float)], Callable[[(float)], (float)]]
    boundary_function : Callable[[(float)], (float)]
    point_sources : ("pointSource")
    domain_conditions : ("DomainCondition")
    # inherited values
    def __init__(self, name, equation, domain, operator, boundary_function=None, point_sources=None):
        if boundary_function==None:
            boundary_function = lambda x: jnp.array([0.,])
        if point_sources==None:
            point_sources = tuple()
        self.equation = equation
        self.domain = domain
        self.operator = operator
        self.boundary_function = boundary_function
        self.point_sources = point_sources

        # find the domain conditions and init superclass
        internal = DomainCondition(self.domain, self.operator)
        boundary = dirichlet_boundaries(self.domain, self.boundary_function)
        points = self.point_sources
        domain_conditions = (internal,) + tuple(boundary) + points
        super().__init__(name = name, domain_conditions = domain_conditions)

def generate_helmholtz_problem(domain, k=1, f=None, point_sources = None):
    operator = generate_helmholtz_operator(k,f)
    return DirichletProblem(name="Helmholtz Problem", equation="\\nabla u + k*u = 0", domain=domain, operator=operator, point_sources = point_sources)
