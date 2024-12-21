## CLS-MML

This is the implementation of our paper: Causal Location-Scale Noise Models by Minimum Message Length.



## Experiments
- To execute the experiments, run run.py after updating the file path to specify the location of your dataset.
- Experiment results are saved in the `/results` directory

## Hyperparameter intervals of nu (degrees of freedom).

We set the hyperparameter nu for each dataset in this way:

Multi, Net, Cha and Tuebingen : [200, 1000];

AN, AN-s, LS, LS-s and MN-U: [100, inf];

SIM, SIM-ln, SIM-c, SIM-G: [10, inf].

## 🛠️ **Dependencies**

- Python 3.x  
- NumPy  
- SciPy  

*(Installed via `install.sh`)*





