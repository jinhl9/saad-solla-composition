# Compositional learning

## Project overview
The aim of the project is to model and analyze compositional learning in neural networks to get insights of the importance of primitives in hierarchical RL. 
We model a compositional RL in teacher-student setup and derive determinisitic on-line gradient learning dynamics in sets of ODEs of order parameters of the neural network using techniques from statistical physics ([Patel et al.(2023)](https://arxiv.org/abs/2306.10404); [Gardner & Derrida (1989)](https://iopscience.iop.org/article/10.1088/0305-4470/21/1/031); [Saad & Solla (1995)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.74.4337); [Biehl & Schwartz (1995)](https://iopscience.iop.org/article/10.1088/0305-4470/28/3/018); [Seung et al. (1996)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056); [Zdeborovà & Krzakala (2016)](https://arxiv.org/abs/1511.02476)).

We analyze learning dynamics in two learning paradigms - primitives pre-training curricula & vanilla learning -, and show the importance of primitives learning in compositional RL and characterize the learning time in relation to the task parameters.   

### Paper: 
[Why Do Animals Need Shaping? A Theory of Task Composition and Curriculum Learning., JH. Lee, S. Sarao Mannelli, A. Saxe](https://arxiv.org/abs/2402.18361), ICML 2024

## Codebase

While the main body of the paper being mostly theory, this codebase was used to run the simulations in simple neural network and follow the numerical ODEs evolution. 


