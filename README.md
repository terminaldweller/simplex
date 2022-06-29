[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5fd619053adf4ce88c4333e306aafa4a)](https://www.codacy.com/gh/terminaldweller/simplex/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=terminaldweller/simplex&amp;utm_campaign=Badge_Grade)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/terminaldweller/simplex.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/terminaldweller/simplex/alerts/)

# Simplex

A python package that solve linear programming problems using the simplex method.<br/>
Features:<br/>
* The Problem is input into the program by a file containing python expression.<br/>
* Solves both min and max problems(duh!).<br/>
* Uses the big M method to find a basic feasible solution when there are none available in the original program.<br/>
* Handles adding slack variables to convert the problem into standard form.<br/>
* Uses the lexicographic rule to prevent ending up in a loop due to degenerate extreme points.<br/>

Run Help to get a list of available commandline options.<br/>
```sh
usage: simplex.py [-h] [--equs EQUS] [--slack SLACK] [--aux AUX] [--iter ITER]
                  [--min] [--verbose] [--debug] [--numba]

options:
  -h, --help            show this help message and exit
  --equs EQUS, -e EQUS  the file containing the equations
  --slack SLACK, -s SLACK
                        slack variable base name, names are cretedby adding a
                        number to the string
  --aux AUX, -a AUX     aux variable base name, names are cretedby adding a
                        number to the string
  --iter ITER, -i ITER  maximum number of iterations
  --min, -m             determines whether its a minimization problem.if not,
                        its a maximization problem
  --verbose, -v         whether to print output verbosely
  --debug, -d           whether to print debug info
```
Example usage:<br/>
```sh
./simplex.py -e ./tests/equ6.py -a xa -v -s z -m
```

## How to Get
You can get it from [pypi](https://pypi.org/project/dsimplex/):<br/>
```sh
pip3 install dsimplex
```
Or you can clone this repo and run it like that:<br/>
```sh
git clone https://github.com/terminaldweller/simplex && cd simplex && poetry install
```

## TODO
* Use numba
