import visdom
vis = visdom.Visdom()
import torch
import random


#vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

for i in range(300):
    vis.line(
        X= [i+25], Y=[i+50], update='append', win='mywin', opts={'yaxis':{
        'range':[25, 300]}}
    )
