"""
Filename: solver.py
Author: Aoibh Schumann
Date: 2025-10
Version: 1.0
Description:
    Find parameters for a neural network to fulfill a DomainConditionProblem.
    Additionally graph the solution (in two dimensions) and generate an history
    of the training
"""

from jaxopt import OptaxSolver
from optax import adam, cosine_decay_schedule
from tqdm import trange

from domains import *
import matplotlib.pyplot as plt
from point_generation import *

def generate_loss(nn, problem, training_points): #-> loss(parameters)
    """
    returns a loss function of network parameters and points

    transforms the abstract loss function which takes a function u from
    domain to codomain and modifies it so it takes the parameters that
    define that function instead.

    This allows for the optimizer to kknow what values to optimize over
    """

    def concrete_loss(params):
        u = nn(params = params)
        output_loss = problem.loss(u, training_points)
        return output_loss

    return concrete_loss


def train(nn: "NN",
          problem: "DomainConditionProblem",
          refinement: int,
          key,
          params=None,
          nIter=5 * 10**4):
    """
    trains the given neural net to obey the given problem
    """

    print("training a solution to " + problem.equation)
    domain_conditions = problem.domain_conditions

    print ("   generating training points...")
    training_points = generate_problem_points(problem=problem, refinement=refinement)


    print ("   generating loss function...")
    loss = generate_loss(nn, problem, training_points) # params |-> loss

    if params==None:
        params = initialize_parameters(key, nn.parameter_geometry)

    lr = cosine_decay_schedule(1e-03, nIter)
    optimizer = OptaxSolver(fun=loss, opt=adam(lr))
    state = optimizer.init_state(params)

    loss_log = []
    for it in (pbar := trange(1, nIter + 1)):
        params, state = optimizer.update(params, state)
        if it % 100 == 0:
            loss = state.value
            loss_log.append(loss)
            pbar.set_postfix({"pinn loss": f"{loss:.3e}"})

    return params, loss_log


### Display Results
#def draw_training(loss_log, save=True):
#    #TODO: FIX THIS BULLSHIT
#    print("Drawing loss log...")
#    dir = f"figures/{self.name}"
#    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
#    # loss log
#    ax1.semilogy(self.loss_log, label="PINN Loss")
#    ax1.set_xlabel("100 iterations")
#    ax1.set_ylabel("Mean Squared Error")

def draw_solution(u, bounds, save=True, name=None, resolution = 100):
    # Solution profile
    assert len(bounds) == 2
    assert len(bounds[0]) == 2
    assert len(bounds[1]) == 2

    domain = Domain(dimension = 2, bounds=bounds)
    points = generate_grid_points(domain, resolution)
    u_points = u(points).reshape((resolution,resolution))
    plt.imshow(u_points, cmap='jet', interpolation='nearest')
    plt.show
    plt.savefig('uplot.png')
    print("Done!")


def draw_loss(u, loss, bounds, save=True, name=None, resolution = 100):
    # Solution profile
    assert len(bounds) == 2
    assert len(bounds[0]) == 2
    assert len(bounds[1]) == 2

    domain = Domain(dimension = 2, bounds=bounds)
    points = generate_grid_points(domain, resolution)
    u_points = loss(u)(points).reshape((resolution,resolution))
    plt.imshow(u_points, cmap='jet', interpolation='nearest')
    plt.show
    plt.savefig('uplotloss.png')
    print("Done!")
