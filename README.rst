This is the code used for the "Optimal modularity and memory capacity of neural networks" paper. The following dependencies are required to run the code:

* Python 2.7
* https://github.com/Nathaniel-Rodriguez/utilities.git
* https://github.com/Nathaniel-Rodriguez/echostatenetwork.git
* https://github.com/Nathaniel-Rodriguez/graphgen.git
* networkx 1.11
* matplotlib 1.5.3
* numpy 1.11.2
* scipy 0.18
* joblib 0.11

Many of the more computationally intensive tasks such as generating the simulations for the contour diagrams used special scripts made for the supercomputers at IU. Some of these scripts are under the bigred2 folders. For local parallel computing the python library joblib was used. With the exception of graphgen, all the packages can be installed using pip. Graphgen can also be installed using pip but requires additional C++ executable files for generating LFR graphs. Details are available in the graphgen repository.