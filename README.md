# Hyperlattice-Queueing-Model

We present a generalized hypercube queue model, building upon the original model by Larson, with a particular focus on its application to overlapping service regions such as police beats. The traditional hypercube queue model is constrained to light-traffic operations, where officers are either ''busy'' or ''not busy''. However, contemporary service operations often experience saturation, necessitating the inclusion of heavy-traffic operations with queue lengths greater than one. The motivation to address the heavy-traffic regime stems from the increased workload and staff shortages prevalent in modern service systems. The design of overlapping regions is inspired by boundary effects in crime incidents, which require overlapping patrol regions for efficient resource allocation. Our proposed model addresses these issues using a Markov model with a large state space represented by integer-valued vectors. By leveraging the sparsity structure of the transition matrix, where transitions occur between states whose vectors differ by 1 in the $\ell_1$ distance, we can accurately solve the steady-state distribution of states. This solution can then be used to evaluate general performance metrics for the service system. We demonstrate the reasonable accuracy of our model through simulation.

[Shixiang Zhu, Wenqian Xing, and Yao Xie. Generalized Hypercube Queuing Models with Overlapping Service Regions.](https://arxiv.org/abs/2304.02824)
