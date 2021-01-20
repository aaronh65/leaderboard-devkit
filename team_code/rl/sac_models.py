from stable_baselines import SAC

class SAC_LB(SAC):
    def __init__(self, policy, env):
        super(SAC_LB, self).__init__(policy, env)
