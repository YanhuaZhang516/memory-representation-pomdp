# memory-representation-pomdp

## 1. [The Enviroment](https://github.com/YanhuaZhang516/memory-representation-pomdp/tree/main/gym_test)
### 1. [gridworld2.py](https://github.com/YanhuaZhang516/memory-representation-pomdp/blob/main/gym_test/gridworld2.py)
It includes two environments:  
- **GridWorldEnv(MDP)**  
**observation_space** = spaces.Discrete(self.n_height * self.n_width)  
**action_space** = spaces.Discrete(4)  
**reward**: -1 for the default Grid; 0 for the goal Grid.

- **GridWorldEnvNew(POMDP)**  
**observation_space** = spaces.Discrete(9)  
**reward**: -1 for the default Grid; 0 for the goal Grid

There are 9 different observations:   
  
        0: the default grid
        1: the wall grid  

        obs[0]: [0 0 0 , 0 0 0, 0 0 0]
        obs[1]: [0 0 1, 0 0 1, 0 0 1]
        obs[2]: [0 0 1, 0 0 1, 1 1 1]
        obs[3]: [0 0 0, 0 0 0, 1 1 1]
        obs[4]: [1 0 0, 1 0 0, 1 1 1]
        obs[5]: [1 0 0, 1 0 0, 1 0 0]
        obs[6]: [1 1 1, 1 0 0 , 1 0 0]
        obs[7]：[1 1 1, 0 0 0, 0 0 0]
        obs[8]: [1 1 1, 0 0 1, 0 0 1]

        return: the index of obs(in the range of (0,8))

### 2. [gridworldRNN](https://github.com/YanhuaZhang516/memory-representation-pomdp/blob/main/gym_test/gridworldRNN.py)
(the child class from GridWorldEnvNew)

It includes two environments, both are used for POMDP.
- **GridWorldEnvRnn(POMDP)**  
**observation_space** = spaces.MultiDiscrete(9,4)  
"9" is for the number of observations; "4" is the number of actions

- **GridWorldEnvRnnNew(POMDP)**  
(for the truncated history)  
**observation_space** = spaces.MultiDiscrete([9 for number of observations],[4 for the number of actions])  


