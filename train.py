"""
Script for model training
rlsn 2024
"""
from env import YunEnv,RPSEnv
from agent import Agent, tabular_Q, softmax
import numpy as np
import argparse, random, time, itertools
from tqdm import tqdm




def estimate_reward(env, num_episodes, p1, p2):
    R=0
    for i in range(num_episodes):
        state, info = env.reset(opponent=p2, train=True, perturb=False)
        for t in itertools.count():
            action = p1.step(state, Amask=env.available_actions(state))
            state, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                R+=r
                break
    return R

def exploitability(beta,pi,Ne=300):
    b1 = Agent(beta[0])
    b2 = Agent(beta[1])
    pi1 = Agent(pi[0])
    pi2 = Agent(pi[1])
    R1 = estimate_reward(env, Ne, b1, pi2)
    R2 = estimate_reward(env, Ne, b2, pi1)
    return R1/Ne, R2/Ne



def PSRO(env, num_iters=1000, num_steps_per_iter = 10000, eps=0.1, alpha=0.1, save_interval=5, num_policies=10):
    Qh, Pih = [],[]
    pi = np.ones([num_policies,env.observation_space.n,env.action_space.n])
    pi = np.random.rand(num_policies,env.observation_space.n,env.action_space.n)

    pi = pi/pi.sum(-1,keepdims=True)

    beta = np.copy(pi)
    expls = [1]
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # reset Q
        Q[0] = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[0,-env.n_ternimal:] = 0 # terminal states to 0
        # train for beta
        for opponent_id in tqdm(range(1,num_policies), desc="Iter", position=1, leave=False):
            env.reset(opponent=Agent(pi[opponent_id]), train=True)
            Q[0] = tabular_Q(env, num_steps_per_iter, Q=Q[0], epsilon=eps, alpha=alpha, eval_interval=-1)
            beta[0] = np.eye(env.action_space.n)[Q[0].argmax(-1)]

        # update avg strategy
        eta = max(0.1/niter,0.0001)
        pi += eta*(beta-pi)

        # eval exploitability
        Ne=300
        r1,r2 = exploitability(beta, pi, Ne=Ne)
        expl=r1+r2
        desc = f"eta={round(eta,4)}, expl={round(expl,2)} {pi.mean(0)[0]}|Iter"
        # desc = f"eta={round(eta,4)}, expl={round(expl,2)}|Iter"
        
        pbar.set_description(desc)
        pbar.refresh()

        # save data
        if niter%save_interval==0:
            Qh.append(Q)
            Pih.append(pi)
            expls.append(expl)

        Q = np.roll(Q,shift=-1,axis=0)
        pi = np.roll(pi,shift=-1,axis=0)
        beta = np.roll(beta,shift=-1,axis=0)

    return np.array(Qh)[::-1], np.array(Pih)[::-1], expls

def fictitious_selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1, save_interval=5):
    Qh, Pih = [],[]
    pi = np.ones([2,env.observation_space.n,env.action_space.n])
    pi = np.random.rand(2,env.observation_space.n,env.action_space.n)

    pi = pi/pi.sum(-1,keepdims=True)

    beta = np.copy(pi)
    expls = [1]
    Ne = 1000
    min_d = 0.01
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:        
        # reset Q
        Q = np.random.randn(2, env.observation_space.n,env.action_space.n)*1e-2
        Q[:,-env.n_ternimal:] = 0 # terminal states to 0
        # train the losing agent for beta
        r = estimate_reward(env, Ne, Agent(pi[0]), Agent(pi[1]))
        if abs(r)<1/np.sqrt(Ne):
            # check for policy distance (L2-norm)
            d = np.sqrt(((pi[0]-pi[1])**2).sum())
            if d<min_d and expl<2/np.sqrt(Ne):
                # similar policy, end training
                print(f"early stop with policy distance={d}<{min_d}")
                break
            else:
                # tied, reinitialize one of them and continue
                print(f"@ local equilibrium, reinitialize policy")
                pi[0]=np.random.rand(env.observation_space.n,env.action_space.n)
                pi[0] = pi[0]/pi[0].sum(-1)
                continue
        agent_id = 0 if r<0 else 1
        opponent_id = agent_id-1

        env.reset(opponent=Agent(pi[opponent_id]), train=True)
        Q[agent_id] = tabular_Q(env, num_steps_per_iter, Q=Q[agent_id], epsilon=eps, alpha=alpha, eval_interval=-1)
        beta[agent_id] = np.eye(env.action_space.n)[Q[agent_id].argmax(-1)]

        # update avg strategy
        eta = max(1/niter,0.001)
        pi[agent_id] += eta*(beta[agent_id]-pi[agent_id])
        pi[agent_id] += eta*(pi[opponent_id]-pi[agent_id])

        # eval exploitability
        r1,r2 = exploitability(beta, pi, Ne=Ne)
        expl=r1+r2
        desc = f"eta={round(eta,4)}, expl={round(expl,2)} {pi[:,0].flatten()}|Iter"
        # desc = f"eta={round(eta,4)}, expl={round(expl,2)}|Iter"
        
        pbar.set_description(desc)
        pbar.refresh()

        # save data
        if niter%save_interval==0:
            Qh.append(Q)
            Pih.append(pi)
            expls.append(expl)

    return np.array(Qh)[::-1], np.array(Pih)[::-1], expls

def selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1):
    pi = np.ones([env.observation_space.n,env.action_space.n])
    pi = pi/pi.sum(-1,keepdims=True)


    beta = np.copy(pi)
    expl = 1
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # rl beta strategy
        
        # reset Q
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
        Q[-env.n_ternimal:] = 0 # terminal states to 0

        env.reset(opponent=Agent(pi), train=True)
        Q = tabular_Q(env, num_steps_per_iter, Q=Q, epsilon=eps, alpha=alpha, eval_interval=-1)
        beta = np.eye(env.action_space.n)[Q.argmax(-1)]


        Ne = 1000
        R = estimate_reward(env, Ne, Agent(beta), Agent(pi))
        expl = R/Ne
        eta = 1/niter
        pi += eta*(beta-pi)


        pbar.set_description(f"eta={round(eta,4)}, expl={round(expl,2)} |Iter")
        pbar.refresh()
    return Q, pi

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--num_iters', type=int, help="number of total training iterations", default=1000)
    parser.add_argument('--num_steps_per_iter', type=int, help="number of training steps for each iteration", default=20000)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=0.1)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)

    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    # env = YunEnv()
    env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    # Q,pi = selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.01)
    Q,pi,expls = fictitious_selfplay(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size)
    # Q,pi,expls = PSRO(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size)


    np.save(args.model_file, {"Q":Q,"PI":pi,"expl":expls})

    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))