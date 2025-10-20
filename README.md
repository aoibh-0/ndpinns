# NN-PDE

Originally based on [Siren\_PINNs](https://github.com/jaeminoh/Siren_PINNs) by
jaeminoh. (I was going to push this to the Siren\_PINNs repository, and then
the scope... got away from me.)

A package for generating solutions to high-dimensional partial differential
equations on irregularly-shaped domains using neural networks, in particularly
so-called SInusoidally REctified Neural networks (SIRENs).


## Contents and Structure

The included files are (in alphabetical order):

- conditions.py
- derivatives.py
- domains.py
- neural\_nets.py
- point\_generation.py
- solver.py


Each of these components has the following dependencies (listed from high level
to low level)

- solver (depends on conditions, point generation, and neural nets)
- conditions (depends on derivatives and domains)
- point generation (depends on conditions and domains)
- neural nets (primitive)
- derivatives (primitive)
- domains (primitive)


## TODO

(vaguely in order of importance)

- Create Requirements.txt and other required utility files
- Move tests from bvp.py into a proper pytest enabled file
- Improve test coverage
- Create examples in strangely-shaped domains
- Create more differential operators
- Add support for Neumann and Cauchy problems
- Expand domain pre-images to include spheres, not just k-cells
- Fix loss graphing
