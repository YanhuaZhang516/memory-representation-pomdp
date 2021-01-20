"""
General GridWorld Environment
"""
from typing import List, Tuple

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
from random import random
# from stable_baselines.common.env_checker import check_env
# from stable_baselines import PPO2
# from stable_baselines.common.evaluation import evaluate_policy

class Grid(object):
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: float = 0.0,
                 value: float = 0.0):  # value属性备用
        self.x = x  # 坐标x
        self.y = y
        self.type = type  # 类别值（0：空；1：障碍或边界）
        self.reward = reward  # 该格子的即时奖励
        self.value = value  # 该格子的价值，暂没用上
        self.name = None  # 该格子的名称
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value:{3}".format(self.x,
                                                                    self.y,
                                                                    self.type,
                                                                    self.value,
                                                                    self.name
                                                                    )


class GridMatrix(object):
    '''格子矩阵，通过不同的设置，模拟不同的格子世界环境
    '''

    def __init__(self, n_width: int,  # 水平方向格子数
                 n_height: int,  # 竖直方向格子数
                 default_type: int = 0,  # 默认类型
                 default_reward: float = 0.0,  # 默认即时奖励值
                 default_value: float = 0.0  # 默认价值（这个有点多余）
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x+1,
                                       y+1,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))

    def get_grid(self, x, y=None):
        '''获取一个格子信息
        args:坐标信息，由x，y表示或仅有一个类型为tuple的x表示
        return:grid object
        '''
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]

        assert (xx >= 1 and yy >= 1 and xx <= self.n_width and yy <= self.n_height), "任意坐标值应在合理区间"

        index = (yy-1) * self.n_width + xx-1
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type


class GridWorldEnv(gym.Env):
    '''格子世界环境，可以模拟各种不同的格子世界
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 7,
                 n_height: int = 7,
                 u_size=40,
                 default_reward: float = -1,
                 default_type=0,
                 max_episode_steps=100
                 ):
        self.u_size = u_size  # 当前格子绘制尺寸
        self.n_width = n_width  # 格子世界宽度（以格子数计）
        self.n_height = n_height  # 高度
        self.width = u_size * n_width  # screen width
        self.height = u_size * n_height  # 场景长度
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)
        self.reward = 0  # for rendering
        self.action = None  # for rendering

        # 0,1,2,3 represent left, right, up, down
        self.action_space = spaces.Discrete(4)

        # 观察空间由low和high决定
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        # 坐标原点为左下角，这个pyglet是一致的
        # 通过设置起始点、终止点以及特殊奖励和类型的格子可以构建各种不同类型的格子世界环境
        # 比如：随机行走、汽车租赁、悬崖行走等David Silver公开课中的示例
        self.end = (5, 5)  # 终止格子坐标，可以有多个
        self.start = (2, 2)  # 起始格子坐标，只有一个
        self.types = []  # 特殊种类的格子在此设置。[(3,2,1)]表示(3,2)处值为1
        self.rewards = []  # 特殊奖励的格子在此设置，终止格子奖励0

        self.viewer = None  # 图形接口对象
        self.seed()  # 产生一个随机子

        # set a time limit
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

        # set the wall to the grid
        index = []
        for i in range(self.n_width):
            index.append((i+1, 1, 1))
            index.append((i+1, self.n_width, 1))
            index.append((1, i+1, 1))
            index.append((self.n_width, i+1, 1))
        self.types = index

        self.refresh_setting()

        self.reset()

    def _adjust_size(self):
        '''调整场景尺寸适合最大宽度、高度不超过800
        '''
        pass

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action = action  # action for rendering
        # add some noise here
        r = random()
        if r <= 0.2:
            self.action = self.action_space.sample()

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
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)
        done = self._is_end_state(new_x, new_y)
        ### 这里修改状态
        self.state = self._xy_to_state(new_x, new_y)

        # 提供格子所在信息
        info = {"x": new_x, "y": new_y, "grids": self.grids, "TimeLimit.truncated": False}

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        return self.state, self.reward, done, info



    # 将状态变为横纵坐标
    def _state_to_xy(self, s):
        y: int = int(s / self.n_width) + 1
        x: int = int(s % self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return self.n_width * (y-1) + x
        elif isinstance(x, tuple):
            return self.n_width * (x[1]-1) + x[0]
        return -1  # 未知状态

    def refresh_setting(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        self._elapsed_steps = 0
        return self.state

    # 判断是否是终止状态
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "坐标数据不完整"
            xx, yy = x[0], x[1]
        # change the end:
        # for end in self.ends:
        #     if xx == end[0] and yy == end[1]:
        #         return True
        if xx == self.end[0] and yy == self.end[1]:
            return True
        return False

    # 图形化界面
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # 格子之间的间隙尺寸

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
                        for i in range(self.n_width+1):
                            line = rendering.Line(start = (i*u_size, 0),
                                                  end =(i*u_size, u_size*self.n_height))
                            line.set_color(0.5,0,0)
                            self.viewer.add_geom(line)
                        for i in range(self.n_height):
                            line = rendering.Line(start = (0, i*u_size),
                                                  end = (u_size*self.n_width, i*u_size))
                            line.set_color(0,0,1)
                            self.viewer.add_geom(line)
            '''

            # 绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x+1, y+1) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x+1, y+1):
                        # 给终点方格添加金黄色边框
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == (x+1) and self.start[1] == (y+1):
                        # 给起始格子设置蓝色边框
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x+1, y+1) == 1:  # 障碍格子用深灰色表示
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            # 更新个体位置
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

class GridWorldEnvNew(GridWorldEnv):

    def __init__(self, n_width, n_height, u_size, default_reward, default_type,max_episode_steps):

        super(GridWorldEnvNew, self).__init__(n_width=n_width,
                                              n_height=n_height,
                                              u_size= u_size,
                                              default_reward = default_reward,
                                              default_type=default_type,
                                              max_episode_steps= max_episode_steps)


        self.observation_space = spaces.Discrete(9)
        # set the start and the end observation
        # set a time limit
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

        # set the wall to the grid
        index = []
        for i in range(self.n_width):
            index.append((i+1, 1, 1))
            index.append((i+1, self.n_width, 1))
            index.append((1, i+1, 1))
            index.append((self.n_width, i+1, 1))
        self.types = index
        self.refresh_setting()
        self.reset()

    def _xy_to_obs(self, x, y=None):
        """
        we set 8 different states here:
        0: the default grid
        1: the wall grid
        state[0]: [0 0 0 , 0 0 0, 0 0 0]
        state[1]: [0 0 1, 0 0 1, 0 0 1]
        state[2]: [0 0 1, 0 0 1, 1 1 1]
        state[3]: [0 0 0, 0 0 0, 1 1 1]
        state[4]: [1 0 0, 1 0 0, 1 1 1]
        state[5]: [1 0 0, 1 0 0, 1 0 0]
        state[6]: [1 1 1, 1 0 0 , 1 0 0]
        state[7]：[1 1 1, 0 0 0, 0 0 0]
        state[8]: [1 1 1, 0 0 1, 0 0 1]

        return: the index of new states(in the range of (0,8))

        """
        s0 = list(np.zeros((9, )))
        s1 = [0, 0, 1,
              0, 0, 1,
              0, 0, 1]
        s2 = [0, 0, 1,
              0, 0, 1,
              1, 1, 1]
        s3 = [0, 0, 0,
              0, 0, 0,
              1, 1, 1]
        s4 = [1, 0, 0,
              1, 0, 0,
              1, 1, 1]  # the initial state (1,1)
        s5 = [1, 0, 0,
              1, 0, 0,
              1, 0, 0]
        s6 = [1, 1, 1,
              1, 0, 0,
              1, 0, 0]
        s7 = [1, 1, 1,
              0, 0, 0,
              0, 0, 0]
        s8 = [1, 1, 1,
              0, 0, 1,
              0, 0, 1]
        s = [s0, s1, s2, s3, s4, s5, s6, s7, s8]

        if isinstance(x, int):
            assert (isinstance(y, int)),"complete position info"
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]

        states = np.zeros((9,))
        for i in range(3):
            for j in range(3):
                states[i*3+j] = self.grids.get_type(xx - 1 + j, yy + 1 - i)
        states = list(states)
        #print("the states is:", states)
        index = 0
        for item in s:
            assert index <= 8, "the object is out of the range "
            if (states == item):
                return index
            index += 1
        #print("index is:",index)


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #self.action = action  # action for rendering
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
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        ### 这里修改第二状态并更新两个状态

        self.observation = self._xy_to_obs(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)

        ## reward for observation, internal state
        self.reward = self.grids.get_reward(new_x, new_y)

        ## judge if the process is finished
        done_state = self._is_end_state(new_x, new_y)

        # 提供格子世界所有的信息在info内
        info = {"x": new_x, "y": new_y, "grids": self.grids,"state":self.state,"TimeLimit.truncated": False}

        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = True
            done_state = True

        return self.observation, self.reward, done_state, info

    def reset(self):
        self.state = self._xy_to_state(self.start)
        self.observation = self._xy_to_obs(self.start)  # state[4]: [1 0 0, 1 0 0, 1 1 1]
        self._elapsed_steps = 0

        return self.observation

    def render(self, mode='human', close=False):
        global rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # 格子间间隙

        # initialzie our view
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

        # 绘制格子
        for x in range(self.n_width):
            for y in range(self.n_height):
                v = [(x * u_size + m, y * u_size + m),
                     ((x + 1) * u_size - m, y * u_size + m),
                     ((x + 1) * u_size - m, (y + 1) * u_size - m),
                     (x * u_size + m, (y + 1) * u_size - m)]

                rect = rendering.FilledPolygon(v)
                r = self.grids.get_reward(x, y) / 10
                if r < 0:
                    rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                elif r > 0:
                    rect.set_color(0.3, 0.5 + r, 0.3)
                else:
                    rect.set_color(0.9, 0.9, 0.9)
                self.viewer.add_geom(rect)

                # 绘制边框
                v_outline = [(x * u_size + m, y * u_size + m),
                             ((x + 1) * u_size - m, y * u_size + m),
                             ((x + 1) * u_size - m, (y + 1) * u_size - m),
                             (x * u_size + m, (y + 1) * u_size - m)]
                outline = rendering.make_polygon(v_outline, False)
                outline.set_linewidth(3)

                # 绘制 Observation 的边框
                # v_obs_outline = [(x * u_size + m, y * u_size + m),
                #                  ((x + 3) * u_size - m, y * u_size + m),
                #                  ((x + 3) * u_size - m, (y + 3) * u_size - m ),
                #                  (x * u_size + m, (y + 3) * u_size - m)]
                v_obs_outline = [((x - 1) * u_size + m, (y - 1) * u_size + m),
                                 ((x + 2) * u_size, (y - 1) * u_size + m),
                                 ((x + 2) * u_size, (y + 2) * u_size),
                                 ((x - 1) * u_size + m, (y + 2) * u_size)]

                obs_outline = rendering.make_polygon(v_obs_outline, False)
                obs_outline.set_linewidth(3)

                if self._is_end_state(x, y):
                    # 给终点方格添加金黄色边框
                    outline.set_color(0.9, 0.9, 0)
                    obs_outline.set_color(0.9, 0.9, 0)
                    self.viewer.add_geom(outline)
                    self.viewer.add_geom(obs_outline)


                if x == 1 and y == 1:
                    # 添加起始点方格
                    outline.set_color(0.5, 0.5, 0.8)
                    self.viewer.add_geom(outline)
                    # 添加起始点的九格方框
                    obs_outline.set_color(0.5, 0.5, 0.8)
                    self.viewer.add_geom(obs_outline)

                if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示
                    rect.set_color(0.3, 0.3, 0.3)
                else:
                    pass

        # set the agent part

        # 填充一个矩形
        # self.agent_obs = rendering.make_polygon([((1 - 1) * u_size + m, (1 - 1) * u_size + m),
        #                                       ((1 + 2) * u_size, (1 - 1) * u_size + m),
        #                                       ((1 + 2) * u_size, (1 + 2) * u_size),
        #                                       ((1 - 1) * u_size + m, (1 + 2) * u_size)], True)
        self.agent_obs = rendering.make_polygon([((1 - 1) * u_size , (1 - 1) * u_size ),
                                                 ((1 + 2) * u_size , (1 - 1) * u_size ),
                                                 ((1 + 2) * u_size , (1 + 2) * u_size),
                                                 ((1 - 1) * u_size , (1 + 2) * u_size)], True)
        self.agent_obs.set_color(1, 1, 204/255.)
        self.viewer.add_geom(self.agent_obs)
        self.agent_obs_trans = rendering.Transform()
        self.agent_obs.add_attr(self.agent_obs_trans)

        # update the position of agent
        x, y = self._state_to_xy(self.state)
        self.agent_obs_trans.set_translation( (x -1) * u_size, (y -1)  * u_size)


        # 绘制圆心
        self.agent = rendering.make_circle(u_size / 4, 30, True)
        self.agent.set_color(1.0, 1.0, 0)
        #self.viewer.add_geom(self.agent)
        self.agent_trans = rendering.Transform()
        self.agent.add_attr(self.agent_trans)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def evaluate(model, num_episodes=100):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_episodes: (int) number of episodes to evaluate it
        :return: (float) Mean reward for the last num_episodes
        """
        # This function will only work for a single Environment
        env = model.get_env()
        all_episode_rewards = []
        for i in range(num_episodes):
            episode_rewards = []
            done = False
            obs = env.reset()
            while not done:
                # _states are only useful when using LSTM policies
                action, _states = model.predict(obs)
                # here, action, rewards and dones are arrays
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)

            all_episode_rewards.append(sum(episode_rewards))

        mean_episode_reward = np.mean(all_episode_rewards)
        # print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

        return mean_episode_reward

if __name__ == "__main__":

    env2 = GridWorldEnvNew(n_width=7, n_height=7, u_size=60, default_reward=-1, default_type=0, max_episode_steps=100)
    env2.start = (2, 2)
    env2.end = (5,5)
    # give special reward to the specific grid

    env2.grids.set_reward(env2.end[0], env2.end[1], 0)


    #env2.refresh_setting()

    print(env2.observation)


    # ### set another enviroment
    # env1 = GridWorldEnv(n_width=7, n_height=7, u_size=60, default_reward=-1, default_type=0)
    # env1.start = (2, 2)
    # env1.end = (5,5 )
    # env1.grids.set_reward(env1.end[0], env1.end[1], 0)
    # env1.refresh_setting()


    def test_render(env):
        start_time = time.time()
        for _ in range(1000):
            #env.render()
            # take a random action
            a = env.action_space.sample()
            state, reward, isdone, info = env.step(a)
            print("{0}, {1}, {2},{3}".format(state, a, reward, isdone))
            print(info["state"])
        print('render close')
        end_time = time.time()
        return end_time-start_time


    process_time2 = test_render(env2)

    #print("the time for model1 is {}, the time for model2 is {}".format(process_time1, process_time2))

    # for _ in range(2000):
    #     env.render()
    #     # 这里是随机选取action
    #     a = env.action_space.sample()
    #     observation, reward, isdone, info = env.step(a)
    #     print("{0}, {1}, {2}, {3}".format(observation, a, reward, isdone))

