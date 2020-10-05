import argparse
import os
import gym
from gym import wrappers
import numpy as np
import agents as Agents
from environments import make_stop_loss_env
from utils import plot_learning_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stop Loss Training')
    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=50000,  # ~13Gb
                        help='Maximum size for memory replay buffer')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=1000,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='OHLCStopLossEnv-v0',
                        help='Market environment.\nOHLCStopLossEnv-v0')
    parser.add_argument('-gpu', type=str, default='0',
                        help='GPU Number: 0, 1, 2, 3')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DuelingDDQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    parser.add_argument('-clip_rewards', type=bool, default=False,
                        help='Clip rewards to range -1 to 1')
    parser.add_argument('-render_video', type=bool, default=False,
                        help='Render video of model')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    env = make_stop_loss_env(env_name=args.env)

    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                   epsilon=args.eps,
                   lr=args.lr,
                   input_dims=env.observation_space.shape,
                   n_actions=env.action_space.n,
                   mem_size=args.max_mem,
                   eps_min=args.eps_min,
                   batch_size=args.bs,
                   replace=args.replace,
                   eps_dec=args.eps_dec,
                   chkpt_dir=args.path,
                   algo=args.algo,
                   env_name=args.env)

    if args.load_checkpoint:
        agent.load_models()

    if args.render_video:
        env = wrappers.Monitor(env, 'tmp/video',
                               video_callable=lambda episode_id: True,
                               force=True)

    fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' \
        + str(args.n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    scores_file = fname + '_scores.npy'

    scores, eps_history = [], []
    n_steps = 0
    steps_array = []
    for i in range(args.n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if args.load_checkpoint and n_steps >= 18000:
            break

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
    np.save(scores_file, np.array(scores))