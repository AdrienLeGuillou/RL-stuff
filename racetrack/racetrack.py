import numpy as np

class RaceTrack:
    def __init__(self, course):
        self.v_max = 5
        self.reset(course)

    def _load_course(self, course):
        # flip course upside down so 0,0 is botton left
        self.course = np.flipud(course)
        # 0 are walls
        # 1 is the track
        # 8 are starts
        # 9 are ends
        self.starts = np.where(self.course == 8)
        self.starts = np.transpose(self.starts)

    def reset(self, course):
        self._load_course(course)
        s = np.random.randint(0, len(self.starts))
        self.position =self.starts[s]
        self.velocity = np.array([1, 0])

    def valid_actions(self):
        return [(x, y) for x in range(-1, 2) for y in range(-1, 2)]

    def _update(self, action):
        temp = self.velocity
        self.velocity += action
        self.velocity = np.clip(self.velocity, 0, self.v_max)
        # if both velocity are 0. Set the previous max to 1
        if np.sum(self.velocity) == 0:
            self.velocity[np.argmax(temp)] = 1

        temp = self.position
        self.position += self.velocity
        self.position[0] = np.clip(self.position[0],
                                   0, np.shape(self.course)[0] - 1)
        self.position[1] = np.clip(self.position[1],
                                   0, np.shape(self.course)[1] - 1)

    def step(self, action):
        self._update(action)
        pos = self.course[self.position[0], self.position[1]]
        observation = (self.position, self.velocity)

        if pos == 0:
            done = True
            reward = -1000
        elif pos == 9:
            done = True
            reward = 0
        else:
            done = False
            reward = -1

        return observation, reward, done

    def print_state(self):
        state = self.course.copy()
        state[self.position[0], self.position[1]] = 5
        print(np.array2string(state))

class Agent:
    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((np.prod(np.shape(self.env.course)), 9))
        self.N = np.zeros((np.prod(np.shape(self.env.course)), 9))
        self.D = np.zeros((np.prod(np.shape(self.env.course)), 9))
        self.pi = np.zeros(np.prod(np.shape(self.env.course))

        self.gamma = 0.9
        self.epsilon = 0.1

        self.mu = np.ones((np.prod(np.shape(self.env.course)), 9))

    def _hash_position(self, position):
        m = np.shape(env.course)[1]
        return m*position[0] + position[1]

    def _position_from_hash(self, p_hash):
        m = np.shape(env.course)[1]
        i = p_hash//m
        j = p_hash % m

        return i, j

    def _hash_action(self, action):
        return 3*(action[0] + 1) + action[1] + 1

    def _action_from_hash(self, a_hash):
        i = a_hash // 3 - 1
        j = a_hash % 3 - 1

        return i, j


course = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
    [0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,8,8,8,8,8,8,0,0,0,0,0,0,0,0]])

env = RaceTrack(course)
player = Agent(env)