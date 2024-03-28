"""
Script for model training
rlsn 2024
"""
from env import YunEnv,RPSEnv
from agent import Agent, tabular_Q, softmax
import numpy as np
import argparse, random, time, itertools
from tqdm import tqdm
from scipy.optimize import linprog

def L2norm(x):
    return np.sqrt((x**2).sum())

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
    return re.x

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
    return R/num_episodes

def exploitability(env,beta,pi,Ne=300):
    b1 = Agent(beta[0])
    b2 = Agent(beta[1])
    pi1 = Agent(pi[0])
    pi2 = Agent(pi[1])
    R1 = estimate_reward(env, Ne, b1, pi2)
    R2 = estimate_reward(env, Ne, b2, pi1)
    return R1, R2

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

def PSROrN(env, num_iters=1000, num_steps_per_iter = 10000, eps=0.1, alpha=0.1, save_interval=1, num_policies=20):
    nash=[]
    Pih = []
    Rh = []
    pi = np.ones([num_policies,env.observation_space.n,env.action_space.n])
    pi = np.random.rand(num_policies,env.observation_space.n,env.action_space.n)
    for s in range(env.observation_space.n):
        pi[:,s]*=env.available_actions(s).reshape(1,-1)
    pi = pi/pi.sum(-1,keepdims=True)
    Ne = 500
    expls = [1]
    divs = [0]

    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # compute nash
        R = gamescape(env, pi, Ne)
        nash_p = solve_nash(R)

        # train agents with positive p
        for agent_id in tqdm(range(num_policies), desc="Agent training", position=1, leave=False):
            # if nash_p[agent_id]<=0:
            #     continue
            # reset Q
            Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2
            Q[-env.n_ternimal:] = 0 # terminal states to 0

            # compute opponent strategy constructed by rectified nash
            pi_weights = nash_p*(R[agent_id]>0).astype(int)
            if pi_weights.sum()<=0:
                pi_weights=np.ones_like(pi_weights)
            pi_weights = pi_weights/pi_weights.sum()
            opponent_pi = pi_weights.reshape(-1,1,1)*pi
            opponent_pi = opponent_pi.sum(0)
            env.reset(opponent=Agent(opponent_pi), train=True)
            Q = tabular_Q(env, num_steps_per_iter, Q=Q, epsilon=eps, alpha=alpha, eval_interval=-1)
            beta = np.eye(env.action_space.n)[Q.argmax(-1)]

            # update avg strategy towards beta
            eta = max(0.5/niter,0.001)
            pi[agent_id] += eta*(beta-pi[agent_id])

        # eval exploitability
        nash_pi = nash_p.reshape(-1,1,1)*pi
        nash_pi = nash_pi.sum(0)
        expl=exploitability_nash(env, nash_pi, pi, Ne=Ne)
        div = (nash_p.reshape(1,-1)@np.maximum(R,0)@nash_p.reshape(-1,1))[0,0]
        # desc = f"eta={round(eta,4)}, expl={round(expl,4)} {nash_pi[0]}|Iter"
        desc = f"eta={round(eta,4)}, expl={round(expl,4)}, div={round(div,4)} | Iter"
        
        pbar.set_description(desc)
        pbar.refresh()

        # save data
        if niter%save_interval==0:
            nash.append(nash_pi)
            Pih.append(pi)
            Rh.append(R)
            expls.append(expl)
            divs.append(div)
    data = {
        "nash":np.array(nash)[::-1],
        "pi":np.array(Pih)[::-1],
        "R":np.array(Rh)[::-1],
        "expl":expls,
        "div":divs
    }
    return data

def fictitious_selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.1, save_interval=5):
    Qh, Pih = [],[]
    pi = np.ones([2,env.observation_space.n,env.action_space.n])
    pi = np.random.rand(2,env.observation_space.n,env.action_space.n)

    pi = pi/pi.sum(-1,keepdims=True)

    beta = np.copy(pi)
    expls = [1]
    r1,r2 = -1,-1
    Ne = 1000
    min_d = 0.01
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:        
        # reset Q
        Q = np.random.randn(2, env.observation_space.n,env.action_space.n)*1e-2
        Q[:,-env.n_ternimal:] = 0 # terminal states to 0
        # train the losing agent for beta
        r = estimate_reward(env, Ne, Agent(pi[0]), Agent(pi[1]))
        d = L2norm(pi[0]-pi[1])
        if abs(r)<0.001 and d<min_d:
            # check for policy distance (L2-norm)
            if expl<0.01:
                # similar policy, end training
                print(f"early stop with exploitability={expl}")
                break
            else:
                # tied, reinitialize one of them and continue
                print(f"@ local equilibrium, reinitialize policy")
                r_id = 1 if r1>r2 else 0
                pi[r_id]=np.random.rand(env.observation_space.n,env.action_space.n)
                pi[r_id] = pi[r_id]/pi[r_id].sum(-1)
                continue
        agent_id = 0 if r<0 else 1
        opponent_id = agent_id-1

        env.reset(opponent=Agent(pi[opponent_id]), train=True)
        Q[agent_id] = tabular_Q(env, num_steps_per_iter, Q=Q[agent_id], epsilon=eps, alpha=alpha, eval_interval=-1)
        beta[agent_id] = np.eye(env.action_space.n)[Q[agent_id].argmax(-1)]

        # update avg strategy
        eta = max(1/niter,0.001)
        g_b = beta[agent_id]-pi[agent_id]
        g_pi = pi[opponent_id]-pi[agent_id]
        pi[agent_id] += eta*(g_b+g_pi)

        # eval exploitability
        r1,r2 = exploitability(env, beta, pi, Ne=Ne)
        expl=r1+r2
        # desc = f"eta={round(eta,4)}, expl={round(expl,2)} {pi[:,0].flatten()}|Iter"
        desc = f"eta={round(eta,4)}, expl={round(expl,4)}|Iter"
        
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
    parser.add_argument('--num_iters', type=int, help="number of total training iterations", default=20)
    parser.add_argument('--num_steps_per_iter', type=int, help="number of training steps for each iteration", default=20000)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=0.1)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)
    parser.add_argument('--num_policies', type=int, help="number of policies for PSRO", default=20)


    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    env = YunEnv()
    # env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    # Q,pi = selfplay(env, num_iters=1000, num_steps_per_iter = 20000, eps=0.1, alpha=0.01)
    # Q,pi,expls = fictitious_selfplay(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size)
    data = PSROrN(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size, num_policies=args.num_policies)


    np.save(args.model_file, data)

    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))