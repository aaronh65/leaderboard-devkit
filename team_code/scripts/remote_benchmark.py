#####################
# this script is usually used on a remote cluster

import os, sys, time
sys.path.append('../..')
import yaml
import subprocess
import argparse
import traceback
import psutil
from datetime import datetime
from common.utils import mkdir_if_not_exists

env_map = {'lbc': 'lblbc', 'dspred': 'dspred'}

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, choices=[10,11], default=11)
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--agent', type=str, default='lbc/image_agent', choices=['lbc/image_agent', 'lbc/auto_pilot', 'lbc/privileged_agent', 'common/forward_agent'])
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('-d', '--debug', action='store_true')

# storage
parser.add_argument('-G', '--gpus', type=int, default=1)
parser.add_argument('--data_root', type=str, default='/data')
args = parser.parse_args()

# specific for multiprocessing and running multiple servers
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

# set carla version variables
carla_root = f'/home/aaronhua/CARLA_0.9.{args.version}'
if args.version == 10:
    carla_root = f'{carla_root}.1'
carla_api = f'{carla_root}/PythonAPI/carla'
carla_egg = f'{carla_root}/PythonAPI/carla/dist/carla-0.9.{args.version}-py3.7-linux-x86_64.egg'

try:

    # make save root + log dir
    project_root = "/home/aaronhua/leaderboard-devkit"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f'debug/{date_str}' if args.debug else f'{date_str}'
    save_root = f'{args.data_root}/leaderboard/results/{args.agent}/{suffix}'
    mkdir_if_not_exists(f'{save_root}/logs')
        
    # launch CARLA servers
    carla_procs = list()
    gpus=list(range(args.gpus))
    port_map = {gpu: (get_open_port(), get_open_port()) for gpu in gpus}
    for gpu in gpus:

        # get open world port, trafficmanager port
        wp, tp = port_map[gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        env["DISPLAY"] = ""
        
        # CARLA command
        cmd = f'bash {carla_root}/CarlaUE4.sh --world-port={wp} -opengl &> {save_root}/logs/CARLA_G{gpu}.txt'
        carla_procs.append(subprocess.Popen(cmd, env=env, shell=True))
        print(f'running {cmd}')

    # warm up CARLA servers otherwise things start to hang
    base_timeout = 3
    timeout = max(args.gpus*base_timeout, 10)
    print(f'Opened {len(gpus)} CARLA servers, warming up for {timeout} seconds')
    time.sleep(timeout)

    # agent-specific configurations
    config = {}
    config['project_root'] = project_root
    config['save_root'] = save_root
    config['save_images'] = args.save_images
    config['split'] = args.split
    privileged = False
    conda_env = 'lb'
    agent_path = args.agent.split('/')
    agent_path = os.path.join(agent_path[0], 'src', agent_path[1])
    if args.agent == 'common/forward_agent':
        pass
    elif args.agent == 'lbc/auto_pilot':
        conda_env = 'lblbc'
        privileged = True
        config['save_data'] = False
    elif args.agent == 'lbc/image_agent':
        conda_env = 'lblbc'
        config['weights_path'] = 'team_code/config/image_model.ckpt'
    elif args.agent == 'lbc/privileged_agent':
        conda_env = 'lblbc'
        privileged = True
        config['weights_path'] = 'team_code/config/map_model.ckpt' 
    config_path = f'{save_root}/config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # route paths
    route_dir = f'{project_root}/leaderboard/data/routes_{args.split}'
    routes = [route.split('.')[0] for route in sorted(os.listdir(route_dir)) if route.endswith('.xml')]

    # tracks relevant information for running on cluster
    lbc_procs = [] # one leaderboard process per route
    gpus_free = [True] * len(gpus) # True if gpu can be used for next leaderboard process
    gpus_procs = [None] * len(gpus) # tracks which gpu has which process
    gpus_routes = [-1] * len(gpus) # tracks the route idx each gpu is working on
    routes_done = [False] * len(routes) # tracks which routes are done (can be True/False/'running')

    # main testing loop
    while False in routes_done or 'running' in routes_done:

        # check for finished leaderboard runs
        for i, (free, proc, route_idx) in enumerate(zip(gpus_free, gpus_procs, gpus_routes)):
            # check if gpus has a valid process and if it's done
            if proc and proc.poll() is not None: 
                gpus_free[i] = True
                gpus_procs[i] = None
                gpus_routes[i] = -1
                routes_done[route_idx] = True

        # wait and continue if gpus are busy
        # or if we're waiting for the last runs to finish
        if True not in gpus_free or False not in routes_done:
            time.sleep(10)
            continue
        
        # otherwise run a new leaderboard process on next route
        route_idx = routes_done.index(False)
        route_name = routes[route_idx]
                        
        # make image + performance plot dirs
        if args.save_images:
            images_path = f'{save_root}/images/{route_name}'
            for rep_number in range(args.repetitions):
                mkdir_if_not_exists(f'{images_path}/repetition_{rep_number:02d}')
        save_perf_path = f'{save_root}/plots/{route_name}'
        mkdir_if_not_exists(save_perf_path)

        # setup env
        gpu = gpus_free.index(True)
        wp, tp = port_map[gpu]

        env = os.environ.copy()
        env["CONDA_ENV"] = conda_env
        env["PROJECT_ROOT"] = project_root
        env["SAVE_ROOT"] = save_root
        env["CARLA_EGG"] = carla_egg
        env["CARLA_API"] = carla_api
        env["HAS_DISPLAY"] = "0"

        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["WORLD_PORT"] = str(wp)
        env["TM_PORT"] = str(tp)
        env["AGENT"] = agent_path
        env["SPLIT"] = args.split
        env["ROUTE_NAME"] = route_name
        env["REPETITIONS"] = str(args.repetitions)
        env["PRIVILEGED"] = str(int(privileged))

        # run command
        cmd = f'bash run_agent.sh &> {save_root}/logs/AGENT_{route_name}.txt'
        print(f'running {cmd} on {args.split}/{route_name} for {args.repetitions} repetitions')
        lbc_procs.append(subprocess.Popen(cmd, env=env, shell=True))

        gpus_free[gpu] = False
        gpus_procs[gpu] = lbc_procs[-1]
        gpus_routes[gpu] = route_idx
        routes_done[route_idx] = 'running'

except KeyboardInterrupt:
    print('detected keyboard interrupt')

except Exception as e:
    traceback.print_exc()

print('shutting down processes...')
for proc in carla_procs + lbc_procs:
    try:
        kill(proc.pid)
    except:
        continue
print('done')
