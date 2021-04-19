import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

from dqn.src.agents.heatmap import ToHeatmap, ToTemporalHeatmap
from misc.utils import *

def get_dqn_actions(Qmap):
    h,w = Qmap.shape[-2:]
    Qmap_flat = Qmap.view(Qmap.shape[:-2] + (-1,)) # (N,T,H*W)
    Q_all, action_flat = torch.max(Qmap_flat, -1, keepdim=True) # (N,T,1)

    action = torch.cat((
        action_flat % w,
        action_flat // w),
        axis=2) # (N,T,2)
    return action, Q_all

hmap = ToTemporalHeatmap(10)

n,c,h,w = (1,4,256,256)

actions = []
for t in range(c):
    actions.append((128, 240-10*t))
actions = torch.Tensor(actions).unsqueeze(0)

input = torch.zeros((n,c,h,w))
input_hmap = hmap(actions, input) 
noise = torch.rand(input_hmap.shape)*0.1
Qmap = input_hmap + noise
noisy_actions, _ = get_dqn_actions(Qmap)
Q_actions = spatial_select(Qmap, actions)
print(Q_actions.numpy().flatten())


off = np.ones_like(actions)*1
#noisy_actions += off
#actions_off1 = actions + off
#actions_off2 = actions - off
#Qo1 = spatial_select(Qmap, actions_off1)
#Qo2 = spatial_select(Qmap, actions_off2)
#print(Qo1.numpy().flatten())
#print(Qo2.numpy().flatten())


images = list()
for t, hmap in enumerate(Qmap[0]):
    hmap_gs = hmap.unsqueeze(0)
    hmap_im = torch.cat((hmap_gs, hmap_gs, hmap_gs), dim=0)
    hmap_im = spatial_norm(hmap_im.unsqueeze(0))[0]*255
    hmap_im = np.uint8(hmap_im.numpy()).transpose(1,2,0)
    hmap_im = Image.fromarray(np.uint8(hmap_im))
    hmap_dr = ImageDraw.Draw(hmap_im)
    x,y = actions[0][t] # 
    hmap_dr.ellipse((x-1,y-1,x+1,y+1), (0,255,0))
    x,y = noisy_actions[0][t]
    hmap_dr.ellipse((x-1,y-1,x+1,y+1), (0,0,255))
    images.append(np.array(hmap_im))

hmap_view = cv2.cvtColor(np.hstack(images), cv2.COLOR_BGR2RGB)
cv2.imshow('hm', hmap_view)
cv2.waitKey(0)
#views_hmap = np.hstack(Qmap[0].numpy())
#cv2.imshow('hmap',views_hmap)
#cv2.waitKey(0)
