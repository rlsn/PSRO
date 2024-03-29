"""
An implementation of tabular Q-learning algorithm
rlsn 2024
"""
import numpy as np
import itertools

class Agent(object):
    def __init__(self, Q, T=1, eps=0.1, mode='prob', thresh=-1e7, name="p1"):
        self.Q = Q
        self.eps = eps
        self.mode = mode
        self.name = name

    def step(self, state, nA=None, Amask=None):
        if nA is None:
            nA = self.Q[state].shape[-1]
        if Amask is None:
            Amask = np.ones(nA)
        if self.Q is None:
            A = np.random.randint(0,nA)
        else:
            if self.mode=='prob':
                p = self.Q[state]*Amask
                if p.sum()>0:
                    A = np.random.choice(nA, p=p/p.sum())
                else:
                    A = np.random.choice(np.arange(nA)[Amask.astype(bool)])
            elif self.mode=='argmax':
                A = epsilon_greedy_policy(self.Q, 0, state, nA)
            elif self.mode=='eps_greedy':
                A = epsilon_greedy_policy(self.Q, self.eps, state, nA)
            else:
                raise NotImplementedError(f"{self.mode} not implemented")
        return A

def epsilon_greedy_policy(Q, epsilon, state, nA):
    if np.random.rand()<epsilon:
        A = np.random.randint(0,nA)
    else:
        A = np.argmax(Q[state][:nA])
    return A

def tabular_Q(env, num_steps, Q=None, discount=0.9, epsilon=0.1, alpha=0.5, eval_interval=1000, n_ternimal=1):
    if Q is None:
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-n_ternimal:] = 0 # terminal states to 0
    i_steps = 0
    acc_return = 0
    acc_length = 0
    for i_episode in itertools.count():
        if i_steps > num_steps:
            break

        if eval_interval>0 and i_episode % eval_interval == 0:
            print("Step {}/{}, Episode {}, avg return = {}, avg length = {}".format(i_steps, num_steps, i_episode, acc_return/eval_interval, acc_length/eval_interval))

            acc_return = 0
            acc_length = 0

        G = 0
        state, info = env.reset()

        for t in itertools.count():
            i_steps += 1

            action = epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)

            next_state, reward, terminated, truncated, _ = env.step(action)

            Q[state, action] += alpha*(reward + discount*Q[next_state].max()-Q[state, action])
            state = next_state

            G = discount*G + reward
            if terminated or truncated:
                acc_length+=t
                acc_return+=G
                break
                
    return Q  

def test():
    import gymnasium as gym
    env = gym.make('CliffWalking-v0', max_episode_steps=50)


    # train
    train = 1
    if train:
        num_steps = 1000000
        Q = tabular_Q(env, num_steps, discount=1.0, epsilon=0.1, alpha=0.5, eval_interval=10000,n_ternimal=1)

        np.save('Q.npy', Q)
            
    else:
        Q = np.load('Q.npy')

    # eval
    env = gym.make('CliffWalking-v0', max_episode_steps=50, render_mode="human")

    observation, info = env.reset(seed=42)
    env.render()
    while(True):
        action = np.argmax(Q[observation])

        observation, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            break

if __name__=="__main__":
    test()