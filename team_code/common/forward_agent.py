from leaderboard.autoagents import autonomous_agent
from carla import VehicleControl

def get_entry_point():
    return 'ForwardAgent'

class ForwardAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        pass
        
    def sensors(self):
        return []
    
    def run_step(self, input_data, timestamp):
        control = VehicleControl()
        control.throttle = 0.7
        control.steer = 0
        control.brake = False
        return control

