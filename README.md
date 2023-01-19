# Matlatzinca
Matlatzinca is GUI for dependence modelling. The program can be used to schematize a Non-Parametric Bayesian Network and define the (conditional) rank correlations. The results is a correlation matrix, that can be used as dependence model. The program is suitable for expert elicitation, as it (i) provided a simple interface for defining a BN, and (ii) helps the user with the mathematical details of defining a valid correlation matrix.

The software is based on the python module py_banshee, which can be found [here](https://github.com/mike-mendoza/py_banshee). Note that this module is described in https://doi.org/10.1016/j.softx.2022.101279, and is based itself on an MATLAB version of the code.

## Usage
To use the program, make sure you have a suitable Python environment (see below), download the repository and double click the run.bat file.

## Python version
Tested for Python 3.8+. Uses the modules numpy, matplotlib, PyQt5, and pydantic. Pydantic is probably the only module you would have to install on top of a standard Anaconda python installation.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
