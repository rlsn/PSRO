"""
Script for model training
rlsn 2024
"""
from env import RPSEnv
from agent import Agent, tabular_Q
import numpy as np
import argparse, time, itertools
from tqdm import tqdm
from scipy.optimize import linprog

def solve_nash(R_matrix):
    A_ub = R_matrix
    D=A_ub.shape[0]
    b_ub = np.zeros(D)
    A_eq = np.zeros([D,D])
    b_eq = np.zeros(D)
    A_eq[0,:]=1
    b_eq[0]=1
    c=np.ones(D)
    re=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    nash_p = np.maximum(re.x,0) # just to make sure non-negative weights
    return nash_p

def estimate_reward(env, num_episodes, p1, p2):
    R=0
    for i in range(num_episodes):
        state, info = env.reset(opponent=p2, train=True)
        for t in itertools.count():
            action = p1.step(state, Amask=env.available_actions(state))
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R+=r
                break
    return R/num_episodes

def exploitability_nash(env,nash_pi,pi,Ne=300):
    R = 0
    nash_agent = Agent(nash_pi)
    for i in tqdm(range(pi.shape[0]), desc="Computing exploitability",position=1,leave=False):
        R+=max(estimate_reward(env, Ne, Agent(pi[i]), Agent(nash_pi)),0)
    return R/pi.shape[0]

def gamescape(env, pi, Ne):
    R = np.zeros([len(pi),len(pi)])
    for i in tqdm(range(len((pi))), desc="Computing gamescape",position=1,leave=False):
        for j in range(len(pi)):
            if j<=i:
                R[i,j] = -R[j,i]
                continue
            R[i,j] = estimate_reward(env,Ne,Agent(pi[i]),Agent(pi[j]))
    return R

def PSRO_Q(env, num_iters=1000, num_steps_per_iter = 10000, eps=0.1, alpha=0.1, save_interval=1, evaluation_episodes=10):
    # initialize a random pure strategy
    tmp = np.random.rand(env.observation_space.n,env.action_space.n)*env.action_matrix
    pi = np.eye(env.action_space.n)[tmp.argmax(-1)]
    pi = np.expand_dims(pi,0)
    expls = [1]
    divs = [0]
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # compute nash
        R = gamescape(env, pi, evaluation_episodes)
        nash_p = solve_nash(R)

        # eval exploitability
        nash_pi = nash_p.reshape(-1,1,1)*pi
        nash_pi = nash_pi.sum(0)
        expl = exploitability_nash(env, nash_pi, pi, Ne=evaluation_episodes)
        div = (nash_p.reshape(1,-1)@np.maximum(R,0)@nash_p.reshape(-1,1))[0,0]

        # train a new agent

        # reset Q
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-env.n_ternimal:] = 0 # terminal states to 0

        env.reset(opponent=Agent(nash_pi), train=True)
        Q = tabular_Q(env, num_steps_per_iter, Q=Q, epsilon=eps, alpha=alpha, eval_interval=-1)
        beta = (Q-Q.min(-1,keepdims=1)+1)*env.action_matrix #to mask out non-actions
        beta = np.eye(env.action_space.n)[beta.argmax(-1)]

        # check criteria for early stopping
        stop=0
        for pi_i in pi:
            if (pi_i == beta).all():
                print("strategy exhausted, early stopping")
                stop=1
                break
        if stop:
            break
        # append strategy
        pi = np.concatenate([pi,np.expand_dims(beta,0)],0)

        desc = f"expl={round(expl,4)}, div={round(div,4)}, nash={nash_pi[0]}| Iter"
        
        pbar.set_description(desc)
        pbar.refresh()

        # save data
        if niter%save_interval==0:
            expls.append(expl)
            divs.append(div)
    data = {
        "nash":nash_pi,
        "pi":pi,
        "R":R,
        "expl":expls,
        "div":divs
    }
    return data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--num_iters', type=int, help="number of total training iterations", default=5)
    parser.add_argument('--num_steps_per_iter', type=int, help="number of training steps for each iteration", default=100)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=0.1)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)

    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    data = PSRO_Q(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size)

    np.save(args.model_file, data)

    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))