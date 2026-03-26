import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SistaHealthEnv(gym.Env):
    TOPICS=['FGM Complications','VVF Causes','Cultural Barriers',
        'Early Marriage','TBA Dangers','Contraception',
        'STIs and HIV','Antenatal Care','Postpartum Care']
    LANGUAGES=['English','Yoruba','Pidgin']
    DOMAINS=['Sexual Health','Maternal Health']
    ACTIONS=['Text Response','Voice Note','Resource Link','Clarify']
    def __init__(self,render_mode=None):
        super().__init__()
        self.observation_space=spaces.Box(
            low=np.array([0,0,0,0,0],dtype=np.float32),
            high=np.array([2,1,8,2,9],dtype=np.float32),dtype=np.float32)
        self.action_space=spaces.Discrete(4)
        self.render_mode=render_mode
        self.state=None; self.step_count=0
        self.episode_reward=0; self.last_action=None
        self.last_reward=0; self.last_feedback=''
    def _get_obs(self): return self.state.astype(np.float32)
    def _get_info(self):
        return {'language':self.LANGUAGES[int(self.state[0])],
            'domain':self.DOMAINS[int(self.state[1])],
            'topic':self.TOPICS[int(self.state[2])],
            'literacy':['Low','Medium','High'][int(self.state[3])],
            'step':int(self.state[4]),'episode_reward':self.episode_reward}
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.state=np.array([self.np_random.integers(0,3),
            self.np_random.integers(0,2),self.np_random.integers(0,9),
            self.np_random.integers(0,3),0],dtype=np.float32)
        self.step_count=0; self.episode_reward=0
        self.last_feedback='New user session started'
        return self._get_obs(),self._get_info()
    def step(self,action):
        assert self.action_space.contains(action)
        literacy=int(self.state[3]); language=int(self.state[0]); step=int(self.state[4])
        reward=0; feedback=''
        if action==0:
            if literacy==2: reward,feedback=10,'Text excellent for high literacy. +10'
            elif literacy==1: reward,feedback=5,'Text good for medium literacy. +5'
            else: reward,feedback=-4,'Text poor for low literacy. -4'
        elif action==1:
            if literacy==0:
                reward,feedback=14,'Voice perfect for low literacy. +14'
                if language==2: reward,feedback=16,'Voice+Pidgin for low literacy. +16'
            elif literacy==1:
                reward,feedback=7,'Voice good for medium literacy. +7'
                if language==2: reward,feedback=9,'Voice+Pidgin for medium literacy. +9'
            else: reward,feedback=2,'Voice acceptable for high literacy. +2'
        elif action==2: reward,feedback=3,'Resource link shared. +3'
        elif action==3:
            if step<3: reward,feedback=4,'Clarification helpful early. +4'
            else: reward,feedback=-2,'Clarification stalling. -2'
        self.state[4]=min(self.state[4]+1,9)
        self.step_count+=1; self.last_action=action
        self.last_reward=reward; self.last_feedback=feedback
        self.episode_reward+=reward
        terminated=self.step_count>=10
        return self._get_obs(),reward,terminated,False,self._get_info()
    def render(self): pass
    def close(self): pass
