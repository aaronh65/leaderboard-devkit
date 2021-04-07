import os, time, argparse, traceback
import subprocess, psutil 
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--debug', action='store_true')
parser.add_argument('-G', '--gpus', type=int, default=1)
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training'])
parser.add_argument('--data_root', type=str, default='/data/aaronhua')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_debug', action='store_true')
parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
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

suffix = f'debug/{args.id}' if args.debug else args.id
save_root = Path(f'{args.data_root}/leaderboard/data/rl/dspred/{suffix}')
save_root.mkdir(parents=True, exist_ok=True)
(save_root / 'logs').mkdir()

try:
    CARLA_ROOT = os.environ['CARLA_ROOT']

    carla_procs = list()
    gpus=list(range(args.gpus))
    port_map = {gpu: (get_open_port(), get_open_port()) for gpu in gpus}
    for gpu in gpus:
        wp, tp = port_map[gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = f'bash {CARLA_ROOT}/CarlaUE4.sh --world-port={wp} -opengl &> {save_root}/logs/CARLA_G{gpu}.txt'
        carla_procs.append(subprocess.Popen(cmd, env=env, shell=True))
        print(f'{cmd}')

    timeout = max(args.gpus*3, 10)
    print(f'opened {len(gpus)} CARLA servers, warming up for {timeout} seconds')
    time.sleep(timeout)

    split_len = {'devtest':4,'testing':26,'training':50,}
    routes = deque(list(range(split_len[args.split])))
    gpu_free = [True] * len(gpus)
    gpu_proc = [None] * len(gpus)

    # routes left or gpus busy
    worker_procs = list()
    while len(routes) > 0 or not all(gpu_free):

        for i, (free, proc) in enumerate(zip(gpu_free, gpu_proc)):
            if proc and proc.poll() is not None:
                gpu_free[i] = True
                gpu_proc[i] = None

        # sleep and goto next iter if busy
        if True not in gpu_free or len(routes) == 0:
            time.sleep(5)
            continue

        # else run the next route
        gpu = gpu_free.index(True)
        wp, tp = port_map[gpu]
        routenum = routes.popleft()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = f'python collect_data.py --data_root {args.data_root} --id {args.id} --split {args.split} --routenum {routenum} --repetitions {args.repetitions} -WP {wp} -TP {tp} --save_data'
        if args.debug:
            cmd += ' --debug'
        if args.save_debug:
            cmd += ' --save_debug'
        worker_procs.append(subprocess.Popen(cmd, env=env, shell=True))
        print(cmd)
        gpu_free[gpu] = False
        gpu_proc[gpu] = worker_procs[-1]

except KeyboardInterrupt:
    print('detected keyboard interrupt')

except Exception as e:
    traceback.print_exc()

print('shutting down processes...')
for proc in carla_procs + worker_procs:
    try:
        kill(proc.pid)
    except Exception as e:
        traceback.print_exc()
        continue
print('done')
