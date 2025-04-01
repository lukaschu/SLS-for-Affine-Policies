# SLS-for-Affine-Policies
This Repo contains the code for our paper on system level synthesis for affine policies.
We compare three nominal MPC formulations
1) Nominal model-based MPC
2) SLS model-based MPC
3) SLS data-driven MPC
   
## Background
We used a simple power grid example with constant coulomb friction. 
Feel free to change the system or the parameters. 

## Build Process
We tested with python3.13, all packages can be found in the requirements.txt file. We propose working in a virtual environment. We show the setup with venv (but of course, feel free to use conda)

1) Setup
  ```
   mkdir Affine_sls_ws && cd Affine_sls_ws
   git clone git@github.com:lukaschu/SLS-for-Affine-Policies.git
  ```
2) Build a virtual environment and download packages
  ```
   python3.13 -m venv env_$name$
  ```
  ```
   source env_$name$/bin/activate
   python -m pip install --upgrade pip
  ```
  ```
   pip install -r requirements.txt
  ```
3) Deploy
  ```
   python main.py
  ```
  Or of course, use your favourite IDE

## License

This code is licensed under the Apache License 2.0. If you use this code in your research, please cite our paper:

Lukas Schüepp, Giulia De Pasquale, Florian Dröfler, "System Level Synthesis for Affine Control Policies: Model-Based and Data-Driven Settings"
TBD

   
