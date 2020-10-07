from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
from market import generate_market_data
from wrapper import *
import gym
import numpy as np
import mplfinance as mpf


class OHLCStopLossEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    """Stop Loss Environment
    Market gym environment with open/high/low/close observation

    The rewards for each step is 1 for positive gain, -1 for negative gain, and 0 for no gain

    The episode terminates after the agent exits or 200 steps have been taken

    Actions:
        Num    Action
        0      exit
        1      1 day low
        2      2 day low
        ...
        30     30 day low
    """

    def __init__(self):
        self.action_space = spaces.Discrete(30)
        self.window_size = 30
        self.current_index = self.window_size-1
        self.max_index = 200
        self.viewer = None
        self.action = 0
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.current_index += 1
        self.action = action

        try:
            currentBar = self.history.iloc[self.current_index]
            prevBar = self.history.iloc[self.current_index - 1]
        except IndexError as e:
            print('trying to index into history at index ',
                  self.current_index, ' history has size', len(self.history))
            print(e)

        reward = prevBar.close - currentBar.close

        if prevBar.stop_loss > currentBar.low or action == 0:
            # need to be able to use normalized values for stop
            reward = prevBar.close - prevBar.stop_loss
            done = True
        else:
            self.history.stop_loss[self.current_index] = self.history.iloc[self.current_index-action+1].low

        start = self.current_index-self.window_size
        end = self.current_index
        state = np.transpose(self.history[['open', 'high', 'low', 'close']][start:end].values)
        return state, reward, done, {}

    def reset(self):
        self.fig = None
        self.ax = None

        # generate history
        self.history = generate_market_data(200, '1/1/2020')
        self.history['stop_loss'] = np.nan
        self.current_index = self.window_size-1

        # intialize to 4 day low
        self.history.stop_loss[self.current_index] = self.history.low[self.current_index -
                                                                      4:self.current_index].min()

        return np.transpose(self.history[['open', 'high', 'low', 'close']][:self.window_size].values)

    def _get_image(self):
        current = self.history.iloc[:self.current_index]
        stop_loss = self.history.iloc[:self.current_index]['stop_loss']
        kwargs = dict(type='candle', volume=False, figscale=2)

        if len(stop_loss.dropna()):
            adp = mpf.make_addplot(stop_loss, type='scatter', color='r', markersize=25)
            self.fig, axes = mpf.plot(current, **kwargs, style='charles',
                                      returnfig=True, addplot=adp)
            self.ax = axes[0]
        else:
            self.fig, axes = mpf.plot(current, **kwargs, style='charles', returnfig=True)
            self.ax = axes[0]

        self.ax.set_title('index={} window size={} action={}'.format(
            self.current_index, self.window_size, self.action))
        self.fig.canvas.draw()

        return cv2.cvtColor(np.array(self.fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2RGB)

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


register(
    id='OHLCStopLossEnv-v0',
    entry_point='environments:OHLCStopLossEnv',
    max_episode_steps=169,
    reward_threshold=0.0,
    # kwargs={'data': change}
)


def make_atari_env(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env


def make_stop_loss_env(env_name, shape=(84, 84, 1)):
    env = gym.make(env_name)
    env = NormalizeReward(env)
    env = PreprocessMarketData(shape, env)

    return env
