#####################
# this script is usually used on a remote cluster

import os, sys, time
import subprocess
import argparse
import traceback
import psutil
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot', 'privileged_agent'])
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--ssd', type=int, default=0, choices=[0,1])
parser.add_argument('--local', action='store_true')
args = parser.parse_args()

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

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

try:

    # make base save path + log dir
    if args.local:
        prefix = '/home/aaron/workspace/carla'
    else:
        prefix = '/home/aaronhua'
    lbc_root = f'{prefix}/2020_CARLA_challenge'
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    if args.debug:
        save_path_base = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
    else:
        save_path_base = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'
    if args.local:
        save_path_base = f'{lbc_root}/{save_path_base}'
    else:
        save_path_base = f'/ssd{args.ssd}/aaronhua/{save_path_base}'

    mkdir_if_not_exists(f'{save_path_base}/logs')
        
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
        cmd = f'bash {prefix}/CARLA_0.9.11/CarlaUE4.sh --world-port={wp} -opengl &> {save_path_base}/logs/CARLA_G{gpu}.txt'
        carla_procs.append(subprocess.Popen(cmd, env=env, shell=True))
        print(f'running {cmd}')

    # warm up CARLA servers otherwise things start to hang
    base_timeout = 3
    timeout = max(args.gpus*base_timeout, 10)
    print(f'Opened {len(gpus)} CARLA servers, warming up for {timeout} seconds')
    time.sleep(timeout)

    # agent-specific configurations
    if args.agent == 'auto_pilot':
        config = 'none' # change to anything except 'none' to save training data
    elif args.agent == 'image_agent':
        config = 'image_model.ckpt' # NN weights in leaderboard/configs
    elif args.agent == 'privileged_agent':
        config = 'map_model.ckpt' # NN weights in leaderboard/configs
    
    # route paths
    route_prefix = f'leaderboard/data/routes_{args.split}'
    routes = [f'{route_prefix}/{route}' for route in sorted(os.listdir(route_prefix)) if route.endswith('.xml')]

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
        route_name = routes[route_idx].split('/')[-1].split('.')[0] # e.g. route_00
                        
        # make image + performance plot dirs
        if args.save_images:
            save_images_path = f'{save_path_base}/images/{route_name}'
            for rep_number in range(args.repetitions):
                mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
        save_perf_path = f'{save_path_base}/plots/{route_name}'
        mkdir_if_not_exists(save_perf_path)

        # setup env
        gpu = gpus_free.index(True)
        wp, tp = port_map[gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        env["LOCAL"] = "1" if args.local else "0"
        env["SAVE_PATH_BASE"] = save_path_base
        env["SAVE_IMAGES"] = "1" if args.save_images else "0"
        env["ROUTE_NAME"] = route_name

        # run command
        cmd = f'bash {lbc_root}/run_leaderboard.sh {wp} {routes[route_idx]} {save_path_base} {tp} {args.agent} {config} {args.repetitions} {prefix} &> {save_path_base}/logs/AGENT_{route_name}.txt'
        print(f'running {cmd}')
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
