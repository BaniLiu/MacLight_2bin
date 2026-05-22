- compare the 2bin model with a traditional RL/SUMO learning (Idea is to show that we have comparative results and the computational time is the same and low in 2bin model learning step)

- Train on in 2bin >test on 1 intersection in sumo  VS Train and test 1 intersection in SUMO 
- Same train in 2 bin>test on 4 intersection in sumo  VS Train and test 4 intersection in SUMO 
  - What to track : Model performances : TTT, ... + Computational time 

- Ensure comparable TSC to 2-bin model, 
  - 2 phases. Phase 1 controls all horizontal movements (EW, EN, ES, WE, WN, WS) and Phase 2 controls all vertical movements (NS, NE, NW, SN, SE, SW). 
    - States = {sum of densities of all incoming lane on phase 1, sum of densities of all incoming lane on phase 2}
  - Action is taken every cycle time (C = 10 sec). Agent chooses the value of green time of phase 1 using RL policy (g1). The value of green time of phase 2, g2 = (C-g1)
- Reward formulation:
  - -(sum of queue length at t+1 - sum of queue length at t) or
  - -(sum of states at t+1 - sum of states at t)
- Pay attention to the demand patterns : 
  - Step 1: Obtain saturation flow rate of intersection (S)
    - In sumo, default demand uses saturation flow rate of corridor S = n*720 veh/hr; n is number of lanes
  - Step 2: Train different RL policies with (i) Low demand: D1 = 0.1*[S,S]; (ii) Medium demand: D2 = 0.4*[S,S]; (ii) High demand: D3 = 0.6*[S,S];  (iv) Mixed demand: D4 = uniform_selection({D1,D2,D3}). The episode starts with a new demand in case of D4.
  - (Goal: To show that even with generalised demand patterns, the agent trained in sumo is unable to learn how to deal with congested state. Intuitively, it should cause earlier gridlock in bigger network.>check generalisabilty across various scenarios (could be checked even within 1 intersection)
- Compute time for learning phases : in the training phase (in sec + GFLOPS metric); 
- time per episod, Nbr or episods to converge 
