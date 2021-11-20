
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')
print(plt.style.available)

c = [0.01,0.01,0.1,0.03]
index = [1]
yawn_time = []
last_unfocus = []
wid = 100


def animate(i):
    data = pd.read_csv('data.csv')
    state = np.array(data['state'])
    yawn = np.array(data["yawn"])
    closed = np.array(data["closed"])
    hands_on_face = np.array(data["hands_on_face"])
    away = np.array(data["away"])
    time = np.array(data["time"])
    new_index = index[-1]-c[0]*hands_on_face[-1]-c[1]*away[-1]

    sec_refocus = (1 - index[-1])*15
    now = [int(i) for i in time[-1].split(":")]
    yawn_time_thresh1=30
    yawn_time_thresh2=1


    if yawn[-1] == 1:
        yawn_time.append(now)
        if len(yawn_time) >= 2:
            t = (yawn_time[-1][0]-yawn_time[-2][0])*3600 + (yawn_time[-1][1]-yawn_time[-2][1])*60  + (yawn_time[-1][2]-yawn_time[-2][2])
            if  t < yawn_time_thresh1 and t> yawn_time_thresh2:
                [int(i) for i in time[-1].split(":")]
                new_index -= c[2]*yawn[-1]

    if state[-1] == 0:
        last_unfocus.append(now)

    if state[-1] == 1:
        if last_unfocus :
            t = (now[0]-last_unfocus[-2][0])*3600 + (now[1]-last_unfocus[-2][1])*60  + (now[2]-last_unfocus[-2][2])
            if t > sec_refocus:
                new_index+=c[3]

    new_index = max(new_index,0)
    index.append(min(1,new_index))
    plt.cla()
    plt.ylabel("Focus index")
    plt.xlabel("time")
    if len(index)<wid:
        if state[-1] == 0:
            plt.plot(range(len(index)), index, label='distracted', color='r',linewidth=2)
            plt.title("distracted",color="r")
        else :
            plt.plot(range(len(index)), index, label='focused', color='g',linewidth=2)
            plt.title("Focused",color="g")
    else:
        if state[-1] == 0:
            plt.plot(list(range(len(index)))[-wid:], index[-wid:], color='r',linewidth=2)
            plt.title("distracted",color="r")
        else :
            plt.plot(list(range(len(index)))[-wid:], index[-wid:], color='g',linewidth=2)
            plt.title("Focused",color="g")
    

    



ani = FuncAnimation(plt.gcf(), animate, interval=10)
#plt.tight_layout()
plt.show()

    

