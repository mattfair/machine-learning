import gym
from gym.spaces import Box
import numpy as np
import pandas as pd
from collections import deque
import cv2
from market import normalize_ohlc


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=False, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)

            if self.clip_reward:
                reward = np.clip(np.array([reward]), 0, 1)[0]

            total_reward += reward

            idx = i % 2
            self.frame_buffer[idx] = obs

            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        info = []
        return max_frame, total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0

        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.event.reset()

        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = observation
        return observation


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(PreprocessFrame, self).__init__(env)

        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        img_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(img_gray, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = Box(env.observation_space.low.repeat(repeat, axis=0),
                                     env.observation_space.high.repeat(repeat, axis=0),
                                     dtype=np.float32)
        self.stack = deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for i in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, obs):
        self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


class PreprocessMarketData(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(PreprocessMarketData, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        df = pd.DataFrame({'open': obs[0], 'high': obs[1], 'low': obs[2], 'close': obs[3]})
        norm = normalize_ohlc(df).values
        resized_screen = cv2.resize(norm, self.shape[1:], interpolation=cv2.INTER_LANCZOS4)
        resized_screen /= resized_screen.max()
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',resized_screen)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return new_obs


class NormalizeReward(gym.Wrapper):
    def __init__(self, env=None, clip_reward=False, normalize_reward=False):
        super(NormalizeReward, self).__init__(env)
        self.clip_reward = clip_reward
        self.normalize_reward = normalize_reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.clip_reward:
            reward = np.clip(np.array([reward]), 0, 1)[0]
        elif self.normalize_reward:
            min = np.min(obs)
            max = np.max(obs)
            diff = max-min
            reward = (reward - min)/diff if diff != 0 else 0

        return obs, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return observation
