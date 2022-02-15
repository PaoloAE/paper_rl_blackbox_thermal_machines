# paper_rl_thermal_machines
Code used to produce the results presented in the manuscript "*Model-free reinforcement-learning of optimal power / efficiency trade-off cycles in quantum thermal machines*" by P.A. Erdman and F. Noé.

## Getting started
To get started, open the [```jupyter_qubit_refrigerator```](jupyter_qubit_refrigerator) or the [```jupyter_harmonic_engine```](jupyter_harmonic_engine) folder if you are interested respectively in the superconducting qubit refrigerator, or in the quantum harmonic oscillator heat engine. Both folders include the following Jupyter Notebooks:
* ```0_train.ipynb```: this Notebook allows to train the agent on the respective quantum thermal machine;
* ```1_produce_pareto.ipynb```: this Notebook loads the agents trained by ```0_train.ipynb```, evaluates their performance, and computes the corresponging Pareto-front. If the data in [```paper_plot_data```](paper_plot_data) was downloaded, it produces exactly the results of the manuscript.
* ```2_paper_plots.ipynb```: this Notebook produces plots in the style of the figures of the manuscript. If the data in [```paper_plot_data```](paper_plot_data) was downloaded, it produces exactly the figures of the manuscript.

[```jupyter_qubit_refrigerator```](jupyter_qubit_refrigerator) contains an additional Notebook called [```3_trapez_cycle_coherence.ipynb```] that computes the average coherence generated in the instantaneous eigen-basis during a trapezoidal cycle.

All values in the Notebooks correspond to the parameters used in the manuscript.

## Acknowledgement
Implementation of the soft actor-critic method based on extensive modifications and generalizations of the code provided at:

J. Achiam, Spinning Up in Deep Reinforcement Learning, https://github.com/openai/spinningup (2018).