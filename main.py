import numpy as np
from util import plot_learning_curve
import os

def run(env, agent, n_games=500):
    best_score = -np.inf
    load_checkpoint = False

    models_dir = 'models/'
    plots_dir = 'plots/'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

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
