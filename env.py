"""
Environment wrapper for agent training
rlsn 2024
"""
import numpy as np
import gymnasium as gym ### pip install gymnasium
from gymnasium import spaces
from agent import Agent

class Rule(object):
    def __init__(self, n_max_energy=5, level=3, init_energy=1) -> None:
            self.n_max_energy = n_max_energy
            self.level = level
            self.n_max_actions = 1 + self.level*2 # 1 yun + n attack + n defense
            self.init_energy = init_energy

            # some stats for analysis/debugging etc.
            self.num_matches = 0 # number of matches since initialization
            self.violation = 0 # attack without enough energy
            self.stupidity = 0 # defense when opponent lacks energy
            self.outpowered = 0 # attack level higher than opponent
            self.busted = 0 # attack when opponent charges
            self.mismatch = 0 # used wrong defense against attack

    def decode_action(self, action_id):
        if action_id is None:
            return 0,0,0
        if action_id == 0: # yun
            return 1, 0, 0
        elif action_id < self.level + 1: # attack
            return 0, action_id, 0
        else: # defense
            return 0, 0 ,action_id-self.level

    def encode_action(self, yun, attack, defense):
        if yun:
            return 0
        elif attack:
            return attack
        else:
            return defense + self.level

    def available_actions(self, s1, s2):
        A = np.zeros(self.n_max_actions)
        A[0]=1
        A[1:min(self.level,s1)+1]=1
        A[self.level+1:min(self.level,s2)+self.level+1]=1
        return A

    def step(self, agent_state:int,
             opponent_state:int,
             agent_action:int,
             opponent_action:int):
        """
        return:
            agent_next_state (left energy)
            opponent_next_state (left energy)
            game_next_state (0 continue 1 opponent win, 2 agent win)
        """
        N = self.n_max_energy + 1 # state dimension
        y1, a1, d1 = self.decode_action(agent_action)
        y2, a2, d2 = self.decode_action(opponent_action)

        # handle states
        agent_next_state = min(agent_state + y1 - a1 , self.n_max_energy)
        opponent_next_state = min(opponent_state + y2 - a2, self.n_max_energy)
        if opponent_next_state<0:
            # game continues with this setup
            opponent_next_state = 0
            a2=0 
        game_next_state = 0

        # punish invalid actions
        if agent_next_state<0:
            agent_next_state = 0
            game_next_state = 1
            self.violation+=1
            self.num_matches+=1
            return agent_next_state, opponent_next_state, game_next_state

        # punish stupid actions
        if d1>opponent_state:
            game_next_state = 1
            self.stupidity+=1
            self.num_matches+=1
            return agent_next_state, opponent_next_state, game_next_state

        # handle result
        if a1 and a1>a2 and a1!=d2: # agent win
            game_next_state = 2
        elif a2 and a2>a1 and a2!=d1: # agent loss
            game_next_state = 1

        if game_next_state>0:
            if y1 or y2:
                self.busted += 1
            elif a1 and a2 and a1!=a2:
                self.outpowered += 1
            else:
                self.mismatch += 1
            self.num_matches+=1
        return agent_next_state, opponent_next_state, game_next_state

class RPSEnv(gym.Env):
    # toy env for testing sanity of the algorithm
    def __init__(self, max_episode_steps=100, **kargs):
        self.opponent = None
        self.train = False
        # Observation is a Cartesian space of the agent's and the opponent's energy,
        # as well as the current game state (ongoing 0/lose 1/win 2)
        self.win_state_id = 1
        self.loss_state_id = 2
        self.n_ternimal = 2
        self.observation_space = spaces.Discrete(3)

        # Action space is the maximum number of actions possible
        self.action_space = spaces.Discrete(3)

        self.max_episode_steps = max_episode_steps

    def _get_info(self):
        return {
            "agent_action":self._agent_action,
            "opponent_action":self._opponent_action,
            "game_state":self._game_state
            }

    def reset(self, seed=None, opponent=None, perturb=True, **kargs):
        super().reset(seed=seed)
        if opponent is not None:
            self.opponent = opponent
        self.perturb=perturb
        # game start
        self._game_state=0

        # value init
        self._agent_action = None
        self._opponent_action = None

        observation = self._game_state
        info = self._get_info()
        self._i_step = 0
        return observation, info
    
    def available_actions(self, state):
        return np.ones(3)

    def step(self, action):
        self._agent_action = action
        if self.opponent is not None:
            self._opponent_action = self.opponent.step(self._game_state, self.action_space.n)
        else:
            self._opponent_action = self.action_space.sample()
        
        d=self._agent_action-self._opponent_action
        bonus = 1
        if d%3==1:
            self._game_state = 1
            reward = 1
            if self._agent_action == 1:
                reward = bonus
        elif d%3==2:
            self._game_state = 2
            reward = -1
            if self._opponent_action==1:
                reward = -bonus
        else:
            self._game_state=0
            reward = 0
        # if self.perturb:
        #     reward+=np.random.randn() # perturbation
        # An episode is done iff one agent has won
        terminated = self._game_state>0
        observation = self._game_state
        info = self._get_info()

        self._i_step+=1
        truncated = self._i_step>=self.max_episode_steps

        return observation, reward, terminated, truncated, info

class YunEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, rule=None, max_episode_steps=20):
        if rule is None:
            rule = Rule()
        self.rule = rule  # The rule or anything informative of the game
        self.opponent = None
        self.train = False
        
        # Observation is a Cartesian space of the agent's and the opponent's energy,
        # as well as the current game state (ongoing 0/lose 1/win 2)
        self.N = rule.n_max_energy + 1 
        self.win_state_id = self.N**2+1
        self.loss_state_id = self.N**2
        self.n_ternimal = 2
        self.observation_space = spaces.MultiDiscrete([self.N,self.N,3])
        self.observation_space.n = self.N**2+2

        # Action space is the maximum number of actions possible
        self.action_space = spaces.Discrete(rule.n_max_actions)

        self.reward_perturbation_std=0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.max_episode_steps = max_episode_steps

    @staticmethod
    def convert_obs(S1,S2,rule):
        n = rule.n_max_energy + 1 
        return min(S1,rule.n_max_energy) * n + min(S2,rule.n_max_energy)

    @staticmethod
    def convert_states(observation, rule):
        n = rule.n_max_energy + 1
        return (observation//n, observation%n)

    def available_actions(self, observation):
        return self.rule.available_actions(*YunEnv.convert_states(observation, self.rule))

    def _get_obs(self):
        # agent's observation as a int
        if self._game_state==0:
            return self._agent_state * self.N + self._opponent_state
        elif self._game_state==1:
            return self.loss_state_id
        else:
            return self.win_state_id

    def _oppo_obs(self):
        # opponent's observation as a int
        if self._game_state==0:
            return self._opponent_state * self.N + self._agent_state
        elif self._game_state==1:
            return self.win_state_id
        else:
            return self.loss_state_id

    def _get_info(self):
        return {
            "agent_action":(self._agent_action,self.rule.decode_action(self._agent_action)),
            "opponent_action":(self._opponent_action,self.rule.decode_action(self._opponent_action)),
            "agent_state":self._agent_state,
            "opponent_state":self._opponent_state,
            "observation":self._get_obs(),
            "game_state":self._game_state
            }
    
    def reset(self, seed=None, opponent=None, train=None, perturb=True):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if train is not None:
            self.train = train
        if opponent is not None:
            self.opponent = opponent
        self.perturb=perturb
        # game start
        self._game_state=0

        # initialize players' energy
        if self.train:
            self._agent_state = np.random.randint(0,self.N)
            self._opponent_state = np.random.randint(0,self.N)
        else:
            self._agent_state = self.rule.init_energy
            self._opponent_state = self.rule.init_energy

        # value init
        self._agent_action = None
        self._opponent_action = None

        observation = self._get_obs()
        info = self._get_info()
        self._i_step = 0
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._agent_action = action
        if self.opponent is not None:
            self._opponent_action = self.opponent.step(self._oppo_obs(), Amask=self.available_actions(self._oppo_obs()))
        else:
            self._opponent_action = self.action_space.sample()
        self._agent_state, self._opponent_state, self._game_state = self.rule.step(agent_state=self._agent_state,
                                                                           opponent_state=self._opponent_state,
                                                                           agent_action=action,
                                                                           opponent_action=self._opponent_action)

        # An episode is done iff one agent has won
        terminated = self._game_state>0
        # rewards
        if self._game_state==2:
            reward = 1
        elif self._game_state==1:
            reward = -1 # binary reward
        else:
            reward = 0
        if self.perturb:
            reward += np.random.randn()*self.reward_perturbation_std #perturbation
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._i_step+=1
        truncated = self._i_step>=self.max_episode_steps


        return observation, reward, terminated, truncated, info

def test():

    env = YunEnv()
    # env = RPSEnv()


    print(env.observation_space.n)
    print(env.action_space.n)

    opponent = Agent(np.random.rand(env.observation_space.n,env.action_space.n))
    agent = Agent(np.random.rand(env.observation_space.n,env.action_space.n))
    observation, info = env.reset(seed=None, opponent=opponent)
    print(0, info)
    for i in range(1,10):
        action = agent.step(observation, Amask=env.available_actions(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        print(i, info)
        if terminated or truncated:
            break

    print("pass")

if __name__=="__main__":
    test()