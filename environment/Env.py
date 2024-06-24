import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

class DoublePoleCartEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,reward_type = "default"):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole1 = 0.1
        self.masspole2 = 0.1
        self.total_mass = self.masscart + self.masspole1 + self.masspole2
        self.length1 = 0.5  # half the pole's length
        self.length2 = 0.5  # half the pole's length
        self.polemass_length1 = self.masspole1 * self.length1
        self.polemass_length2 = self.masspole2 * self.length2
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.reward_type = reward_type

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.reset()

        self.fig, self.ax = None, None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        state = self.state
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        force = self.force_mag if action == 1 else -self.force_mag if action == 0 else 0
        torque = action if action == 2 else 0

        costheta1 = math.cos(theta1)
        sintheta1 = math.sin(theta1)
        costheta2 = math.cos(theta2)
        sintheta2 = math.sin(theta2)

        temp = (force + self.polemass_length1 * theta1_dot ** 2 * sintheta1) / self.total_mass
        theta1acc = (self.gravity * sintheta1 - costheta1 * temp) / (self.length1 * (4.0 / 3.0 - self.masspole1 * costheta1 ** 2 / self.total_mass))
        theta2acc = (self.gravity * sintheta2 + torque - costheta2 * temp) / (self.length2 * (4.0 / 3.0 - self.masspole2 * costheta2 ** 2 / self.total_mass))
        xacc = temp - self.polemass_length1 * theta1acc * costheta1 / self.total_mass - self.polemass_length2 * theta2acc * costheta2 / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta1 = theta1 + self.tau * theta1_dot
        theta1_dot = theta1_dot + self.tau * theta1acc
        theta2 = theta2 + self.tau * theta2_dot
        theta2_dot = theta2_dot + self.tau * theta2acc
        self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta1 < -self.theta_threshold_radians \
               or theta1 > self.theta_threshold_radians \
               or theta2 < -self.theta_threshold_radians \
               or theta2 > self.theta_threshold_radians
        done = bool(done)
        
        if self.reward_type == "default":
            angle_penalty_term = 0.0
        elif self.reward_type == "custom_reward1":
            angle_penalty_term = (abs(theta1) + abs(theta2)) / 90
        elif self.reward_type == "custom_reward2":
            angle_penalty_term = (2 * abs(theta1) + abs(theta2)) / 90
        elif self.reward_type == "custom_reward3":
            angle_penalty_term = (2 * abs(theta1) + 1.5 * abs(theta2)) / 90
        elif self.reward_type == "custom_reward4":
            angle_penalty_term = (abs(theta1) + abs(theta2)) / 90
            angle_penalty_term = angle_penalty_term if angle_penalty_term > 0 else -0.01
        elif self.reward_type == "custom_reward5":
            position_reward = 0.001 * max(0, (self.x_threshold - abs(x)) / self.x_threshold)
            angle_reward = 0.001 * max(0, (self.theta_threshold_radians - (abs(theta1) + abs(theta2)) / 2) / self.theta_threshold_radians)
            velocity_penalty = 0.001 * abs(x_dot)
            angular_velocity_penalty = 0.001 * (abs(theta1_dot) + abs(theta2_dot)) / 2
            angle_penalty_term = angular_velocity_penalty + velocity_penalty - angle_reward - position_reward
        elif self.reward_type == "custom_reward6":
            velocity_penalty = 0.001 * abs(x_dot)
            angular_velocity_penalty = 0.001 * (abs(theta1_dot) + abs(theta2_dot)) / 2
            angle_penalty_term = angular_velocity_penalty + velocity_penalty
        elif self.reward_type == "custom_reward7":
            velocity_penalty = abs(x_dot)**2
            angular_velocity_penalty = ((abs(theta1_dot)**2 + abs(theta2_dot))**2) / 2
            angle_penalty_term = 0.0001 * angular_velocity_penalty + 0.001 * velocity_penalty

        

        if not done:
            reward = 1.0 - angle_penalty_term
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0 - angle_penalty_term
        else:
            self.steps_beyond_done += 1
            reward = 0.0 - angle_penalty_term

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            world_width = self.x_threshold * 2
            scale = screen_width / world_width
            carty = 100  # TOP OF CART
            polewidth = 10.0
            polelen1 = scale * (2 * self.length1)
            polelen2 = scale * (2 * self.length2)
            cartwidth = 50.0
            cartheight = 30.0

            if self.fig is None:
                self.fig, self.ax = plt.subplots()

            self.ax.clear()
            self.ax.set_xlim(-self.x_threshold * scale, self.x_threshold * scale)
            self.ax.set_ylim(0, screen_height)

            x = self.state[0]

            cart = Rectangle((x * scale - cartwidth / 2, carty - cartheight / 2), cartwidth, cartheight, fill=True, color='black')
            self.ax.add_patch(cart)

            pole1_x = x * scale + polelen1 * math.sin(self.state[2])
            pole1_y = carty + polelen1 * math.cos(self.state[2])
            pole1 = plt.Line2D([x * scale, pole1_x],
                            [carty, pole1_y],
                            linewidth=polewidth, color='blue')
            self.ax.add_line(pole1)

            pole2_x = pole1_x + polelen2 * math.sin(self.state[4])
            pole2_y = pole1_y + polelen2 * math.cos(self.state[4])
            pole2 = plt.Line2D([pole1_x, pole2_x],
                            [pole1_y, pole2_y],
                            linewidth=polewidth, color='red')
            self.ax.add_line(pole2)

            plt.pause(0.001)
            plt.draw()
        else:
            return

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None


if __name__ == '__main__':
    env = DoublePoleCartEnv()
    obs = env.reset()
    print(obs)

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break
        
    env.close()
