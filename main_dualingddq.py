from agents import DuelingDoubleDQNAgent
from util import make_env
from main import run

if __name__ == '__main__':
    env_name='PongNoFrameskip-v4'
    env = make_env(env_name, 300)

    # setting mem_size to 20k to run on machine with 8 GB, increase to higher for machines with more
    # memory
    agent = DuelingDoubleDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                  n_actions=env.action_space.n, mem_size=20000, eps_min=0.1, batch_size=32,
                                  replace=1000, eps_dec=1e-5, checkpoint_dir='models/',
                                  algo='DuelingDDQNAgent', env_name=env_name)

    run(env, agent)
