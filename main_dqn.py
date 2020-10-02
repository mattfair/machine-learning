import numpy as np
from agents import DQNAgent
from util import make_env, plot_learning_curve
import os

if __name__ == '__main__':
    env_name='PongNoFrameskip-v4'
    env = make_env(env_name)
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500

    models_dir = 'models/'
    plots_dir = 'plots/'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # setting mem_size to 20k to run on machine with 8 GB, increase to higher for machines with more
    # memory
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=20000, eps_min=0.1, batch_size=32,
                     replace=1000, eps_dec=1e-5, checkpoint_dir='models/', algo='DQNAgent',
                     env_name=env_name)
    print('Using device:', agent.q_eval.device)

    if load_checkpoint:
        agent.load_checkpoint()

    fname = '{}_{}_lr{}_{}games'.format(agent.algo, agent.env_name, agent.lr, n_games)
    figure_file = 'plots/{}.png'.format(fname)

    n_steps = 0
    scores = []
    eps_hist = []
    steps_array = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best score %.1f epsilon %.2f' % 
              (avg_score, best_score, agent.epsilon), 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_hist.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_hist, figure_file)
