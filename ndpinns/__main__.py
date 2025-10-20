from solver import *
from derivatives import *
from conditions import *
from neural_nets import *
from point_generation import *

import time
import jax.random as jr
import json


def keygen():
    epoch_time = int(time.time())
    return jr.key(seed=epoch_time)


test_key = keygen()

R = 10.
BOUNDS = ((-2*R, 2*R), (-2*R, 2*R))
IMGBOUNDS = ((-R, R), (-R, R))

operator = generate_helmholtz_operator(k = 0.05)
point_sources = (PointSource(location= jnp.array((0., 0.)), constant = 1.), )
domain = Domain(dimension=2, bounds = BOUNDS)
problem = generate_helmholtz_problem(domain = domain, point_sources=point_sources)
geometry = ((2,8), (8,1))
siren = Siren( geometry )
params = initialize_parameters(test_key, geometry)



out_params, loss_log = train(nn=siren, problem=problem, nIter = 1000, refinement=50, params=params, key=test_key)

print(out_params)

with open('parameters'+str(time.time())+'.json', 'w') as f:
    json.dump(out_params, f)

sol_u = siren(out_params)
print("initial_loss: " + str(loss_log[0]))
print("  final_loss: " + str(loss_log[-1]))



sol_u0 = lambda x: sol_u(x)[...,0]
sol_u1 = lambda x: sol_u(x)[...,1]
draw_solution(sol_u0, bounds = IMGBOUNDS, save=False)
draw_loss(sol_u0, operator, bounds = IMGBOUNDS, save=False)
