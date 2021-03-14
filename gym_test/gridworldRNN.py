from gridworld2 import *

class GridWorldEnvRnn(GridWorldEnvNew, gym.Wrapper):
    def __init__(self, n_width, n_height, u_size, default_type, max_episode_steps,default_reward):

        super(GridWorldEnvRnn, self).__init__(n_width=n_width,
                                              n_height=n_height,
                                              u_size=u_size,
                                              default_type=default_type,
                                              default_reward=default_reward,
                                              max_episode_steps=max_episode_steps
                                              )
        # Todo
        self.reward = default_reward
        self.action = None
        self.observation = None
        self.input = None
        self.state = None
        # first is action, second is observation
        self.observation_space = spaces.MultiDiscrete([4, 9])

        self.start = (2, 2)
        self.end = (5, 5)

        # find the largest distance between the state and the end state for normalization:
        n_width_max = max([np.abs(i+2 - self.end[0]) for i in range(self.n_width-2)])
        n_height_max = max([np.abs(i+2 - self.end[1]) for i in range(self.n_height-2)])
        self.n_max = n_height_max+n_width_max

        # set a time limit
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None


        # set the wall to the grid:
        index = []
        for i in range(self.n_width):
            index.append((i+1, 1, 1))
            index.append((i+1, self.n_width, 1))
            index.append((1, i+1, 1))
            index.append((self.n_width, i+1, 1))
        self.types = index
        self.refresh_setting()

        self.reset()

    def get_reward(self, x, y):
        """
        we get the reward function through the L1 norm between the current state and end state
        :param state: the current state of the agent
        :return: the reward
        """

        # redefine the end
        l_1 = np.abs(x-self.end[0]) + np.abs(y-self.end[1])
        reward = -1*l_1/self.n_max

        return reward


    def step(self,action):
        assert self._elapsed_steps is not None, "cannot call env.step() before calling reset() "
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        ## to do the step part
        # add some noise here
        r = random()
        if r <= 0.2:
            self.action = self.action_space.sample()
        else:
            self.action = action  # action for rendering
        #  the env store the internal state
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        # boundary effect
        if new_x < 1: new_x = 1
        if new_x > self.n_width: new_x = self.n_width
        if new_y < 1: new_y = 1
        if new_y > self.n_height: new_y = self.n_height


        # wall effect:
        # when the type of grid is 1, it means that the object couldn't get in
        if self.grids.get_type(new_x, new_y) == 1: new_x, new_y = old_x, old_y

        # 修改状态，观测值
        self.observation = self._xy_to_obs(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        # change the whole observation part
        self.input = np.asarray([self.action, self.observation], np.int64)

        # reward for the state(obs)
        self.reward = self.get_reward(new_x, new_y)

        # judge if the process is finished
        done = self._is_end_state(new_x, new_y)

        # 提供格子所在信息
        info = {"x": new_x, "y": new_y, "state":self.state,"grids": self.grids, "TimeLimit.truncated": False}

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        return self.input, self.reward, done, info


    def reset(self):
        self.state = self._xy_to_state(self.start)
        self.observation = self._xy_to_obs(self.start)
        self.action = 0
        self._elapsed_steps = 0
        # Todo
        self.input = np.asarray([self.action, self.observation], np.int64)
        return self.input, self.state

class GridWorldEnvRnnNew(GridWorldEnvRnn):
    def __init__(self, n_width, n_height, u_size, default_type, max_episode_steps,default_reward,num_obs):

        super(GridWorldEnvRnnNew, self).__init__(n_width=n_width,
                                              n_height=n_height,
                                              u_size=u_size,
                                              default_type=default_type,
                                              default_reward=default_reward,
                                               max_episode_steps=max_episode_steps
                                                )


        self.num_obs = num_obs
        lst1 = [9 for i in range(self.num_obs)] + [4 for i in range(self.num_obs)]
        self.observation_space = spaces.MultiDiscrete(lst1)
        self.obs_list = [self.observation for i in range(self.num_obs)]
        self.act_list = [self.action for i in range(self.num_obs)]
        # print("observation space:", self.observation_space)
        # print("the num obs",self.num_obs)


        # find the largest distance between the state and the end state for normalization:
        n_width_max = max([np.abs(i+2 - self.end[0]) for i in range(self.n_width-2)])
        n_height_max = max([np.abs(i+2 - self.end[1]) for i in range(self.n_height-2)])
        self.n_max = n_height_max+n_width_max

        # set the wall to the grid:
        index = []
        for i in range(self.n_width):
            index.append((i+1, 1, 1))
            index.append((i+1, self.n_width, 1))
            index.append((1, i+1, 1))
            index.append((self.n_width, i+1, 1))
        self.types = index
        self.refresh_setting()

        self.reset()

    def step(self, action):
        assert self._elapsed_steps is not None, "cannot call env.step() before calling reset() "
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        ## to do the step part
        # add some noise here
        r = random()
        if r <= 0.2:
            self.action = self.action_space.sample()
        else:
            self.action = action  # action for rendering
        #  the env store the internal state
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        # boundary effect
        if new_x < 1: new_x = 1
        if new_x > self.n_width: new_x = self.n_width
        if new_y < 1: new_y = 1
        if new_y > self.n_height: new_y = self.n_height


        # wall effect:
        # when the type of grid is 1, it means that the object couldn't get in
        if self.grids.get_type(new_x, new_y) == 1: new_x, new_y = old_x, old_y

        # 修改状态，观测值
        self.observation = self._xy_to_obs(new_x, new_y)
        #print("the new obs:", self.observation)
        # store the new observation to the list
        self.obs_list.insert(0, self.obs_list.pop())
        self.obs_list[0] = self.observation
        # store the action to the list
        self.act_list.insert(0, self.act_list.pop())
        self.act_list[0] = self.action

        self.state = self._xy_to_state(new_x, new_y)
        # change the whole observation part
        self.input = np.asarray(self.obs_list+self.act_list, np.int64)

        # reward for the state(obs)
        self.reward = self.get_reward(new_x, new_y)

        # judge if the process is finished
        done = self._is_end_state(new_x, new_y)

        # 提供格子所在信息
        info = {"x": new_x, "y": new_y, "grids": self.grids, "TimeLimit.truncated": False,"observation": self.observation}

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        return self.input, self.reward, done, info


    def reset(self):
        self.state = self._xy_to_state(self.start)
        self.observation = self._xy_to_obs(self.start)
        #print("the observation:",self.observation)
        self.action = 0
        self._elapsed_steps = 0
        # Todo
        self.num_obs=4
        self.obs_list = [self.observation for i in range(self.num_obs)]
        self.act_list = [self.action for i in range(self.num_obs)]
        #self.obs_list = [self.observation, self.observation, self.observation, self.observation]
        #print(self.obs_list)
        self.input = np.asarray(self.obs_list+self.act_list, np.int64)
        return self.input

    def print_obs(self):
        print(self.num_obs)





if __name__ == "__main__":
    # env = GridWorldEnvRnnNew(n_width=7, n_height=7, u_size=60, default_type=0,
    #                          max_episode_steps=200, default_reward=-1, num_obs=4)
    env = GridWorldEnvRnn(n_width=7, n_height=7, u_size=60, default_type=0, max_episode_steps=100,default_reward=-1)

    env.start=(2,2)
    env.end = (5,5)
    env.refresh_setting()

    initial_obs, initial_state = env.reset()

    #env.print_obs()
    list1=[initial_obs]
    list_s=[initial_state]

    start = time.time()
    for _ in range(250):

        a = env.action_space.sample()
        obs, reward, isdone, info = env.step(a)
        #print("{0},{1},{2},{3}".format(obs, a, reward, isdone))
        list1.append(obs)
        list_s.append(info["state"])
        print("[action,observation] for {} is {}".format(_+1, obs))
        print("state is:",info["state"])

        if isdone == True:
            print(info["TimeLimit.truncated"])
    end = time.time()
    print(list1[0][0])
    print(len(list_s))

    print("the process time is:", end-start)











