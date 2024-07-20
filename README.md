# CategorizationStrategies

Code for Collina et al, "Individual-specific strategies in category learning inform learned boundaries"

Data necessary to reproduce figures is available at https://doi.org/10.5061/dryad.73n5tb359

Code is written for Python 3.11.4

Dependencies:

- numpy
- pandas
- matplotlib
- os
- scipy
- pickle
- scikit-learn
- psytrack
- tslearn
- pybads

Guide:

The notebooks Figure1, Figure2, Figure3 and Figure4 generate the panels in the respective paper figures. 

For more detailed demonstrations of the methods used in the paper, analysis_GenerateTrajectories details how we transformed the raw behavioral data using a GLM, analysis_TrajectoryClustering details how we transformed the behavioral data and applied and validated time-series clustering, and analysis_SimulateRLModel details the validation and application of the Choice-History reinforcement learning model used in the paper.

All other files consist of auxiliary functions used for plotting, clustering, simulating and fitting reinforcement learning models, and general data transformation and curation.

The analysis_GenerateTrajectories and analysis_SimulateRLModel notebooks generate data to a created folder titles "new_data", which is then accessed in later parts of the notebooks. Multiple runs of notebook may result in writing over previously saved data.

