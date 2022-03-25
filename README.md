  
Multicell Power Control under QoS Requirements with CNet
===
The trained model and the effects of hyperparameters (e.g., batchsize and learning rate) are stored in model&hyperparameter.zip
The testing data are stored in testing data.zip
===
Abstract
---
Multicell power control for sum rate maximization (SRM) is a widely-studied non-convex resource allocation problem in wireless communication systems. 
Due to the high complexity of the traditional mathematical algorithms, this paper proposes a novel deep neural network (DNN) to solve the SRM problem 
with both QoS and per-base station power requirements. The main challenge here is how to ensure that the power control results learned through DNN 
always meet the QoS requirements with coupled variables. To overcome this challenge, we first transform the feasible set of the SRM problem into its 
V-representation, which can be used in the gradient-based DNN by polyhedron decomposition theorem. With the V-representation of the feasible set,
we directly incorporate the constraints into the DNN to obtain the feasible power control solutions. Simulation results validate that the proposed DNN 
can always meet the requirements and achieve a sum-rate performance close to that optimized by the mathematical algorithm but with much lower computational complexity. 
Moreover, our proposal can also be generalized to solve non-convex problems with linear constraints.

Testing Data
---
The testing date with different number of cells (K) are shown in testing data folder.
