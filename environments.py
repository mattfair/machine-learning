from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
from market import generate_market_data
from wrapper import RepeatActionAndMaxFrame, PreprocessFrame, StackFrames, PreprocessMarketData 
import gym
import numpy as np

class OHLCStopLossEnv(gym.Env):
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
        self.max_index = 200 + self.window_size
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.current_index += 1

        if self.current_index >= self.max_index:
            reward = 0
            done = True
            return np.array([]), reward, done, {}

        currentBar = self.history.iloc[self.current_index]
        prevBar = self.history.iloc[self.current_index - 1]

        start = self.current_index - self.window_size
        end = self.current_index

        if prevBar.stop_loss > currentBar.low or action == 0:
            # need to be able to use normalized values for stop
            reward = 0
            done = True
        else:
            self.history.stop_loss[self.current_index] = self.history.iloc[self.current_index-action+1].low

            diff = self.history.iloc[self.current_index].close - self.history.iloc[self.current_index-1].close
            if diff > 0:
                reward = 1
            elif diff < 0:
                reward = -1
            else:
                reward = 0

        start = self.current_index-self.window_size
        end = self.current_index
        state = np.transpose(self.history[['open', 'high', 'low', 'close']][start:end].values)
        return state, reward, done, {}

    def reset(self):
        # generate history
        self.history = generate_market_data(200, '1/1/2020')
        self.history['stop_loss'] = np.nan
        self.current_index = self.window_size-1

        # intialize to 4 day low
        self.history.stop_loss[self.current_index] = self.history.low[self.current_index -
                                                                      4:self.current_index].min()

        return np.transpose(self.history[['open', 'high', 'low', 'close']][:self.window_size].values)

    def render(self, mode='human'):
        current = self.history.iloc[:self.current_index]
        stop_loss = self.history.iloc[:self.current_index]['stop_loss']
        kwargs = dict(type='candle', volume=False, figscale=1)

        if self.fig is None:
            adp = mpf.make_addplot(
                stop_loss, type='scatter', color='r', markersize=25)
            self.fig, axes = mpf.plot(
                current, **kwargs, style='charles', returnfig=True, addplot=adp)
            self.ax = axes[0]
            # self.ani = animation.FuncAnimation(self.fig, self.render, interval=250)

        self.ax.clear()
        adp = mpf.make_addplot(stop_loss, type='scatter',
                               color='r', markersize=25)
        mpf.plot(current, **kwargs, style='charles', addplot=adp)
        mpf.show()


register(
    id='OHLCStopLossEnv-v0',
    entry_point='environments:OHLCStopLossEnv',
    max_episode_steps=200,
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
    env = PreprocessMarketData(shape, env)

    return env
