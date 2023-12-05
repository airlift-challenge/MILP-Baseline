# ✈️ Airlift Challenge Mixed Integer Linear Program Baseline

This baseline submission is part of the Airlift Challenge,
a competition in which participants must design agents that can plan and execute an airlift operation.
This repository provides an example solution based on Mixed Integer Linear Programs as formulated in the following paper:

   Dimitris Bertsimas, Allison Chang, Velibor V. Mišić, Nishanth Mundru (2019)
   The Airlift Planning Problem.
   Transportation Science.
   53 (3): 773-795.

This code can be submitted to CodaLab.
By default it installs `glpk` open source solver in the Docker evaluation container.
NOTE: As currently implemented, the solution will exceed the time limits, and is such is not viable competition. We are only providing it as a starting point for those who might want to use this approach. 

# Important links
* For more information about the competition: see the [documentation](https://airlift-challenge.github.io/).
* The simulator can be found [here](https://github.com/airlift-challenge/airlift)
* For submissions and to participate in the discussion board: see the [competition platform on CodaLab](https://codalab.lisn.upsaclay.fr/competitions/16103)


## Setup for local evaluation
Inside this repo, do the following:

1) Create the environment
   ```
   conda env create -f environment.yml
   conda activate milp-solution
   ```
2) Install the glpk solver via apt-get:
   ```
   sudo apt-get install glpk-utils
   ```
3) Put a set of scenarios inside a folder named `scenarios`
4) Run
   ```
   python run_custom_scenario.py
   ``` 
