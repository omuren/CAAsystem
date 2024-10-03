import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys
import pickle
from MAsystem import Controller

# PKLファイルのパス
file_path_pkl = 'loc_history.pkl'

# データを読み込む
data = pd.read_pickle(file_path_pkl)
ped_data_all=data['ped_data']
ped_data = ped_data_all[-1]
steer_data = data['steer_data']
time = data['time']
assist_data = data['assist_data']
alpha = [point[0] for point in assist_data]
beta = [point[1] for point in assist_data]

#障害物出現時刻
k = 0
while k < len(ped_data_all):
    if len(ped_data_all[k])>0:
        break
    k += 1

t_start = time[k]
print(t_start)

#平均時間
delta = []
for i in range(k, len(time)-1):
    delta.append(time[i+1] - time[i])
t_ave = sum(delta)/len(delta)
print(t_ave)
print(time[-1])
#print(assist_data)

#保存用
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, 'MAfigures')
file_path = os.path.join(save_dir, 'test0910_avoidance.png')
file_path_2 = os.path.join(save_dir, 'test0910_input.png')
data_file_path = os.path.join(save_dir, '0930_exp2_beta0997.pkl')
if os.path.exists(file_path):
    print(f"error: {file_path} already exists")
    if os.path.exists(file_path_2):
        print(f"error: {file_path_2} already exists")
    if os.path.exists(data_file_path):
        print(f"error: {data_file_path} already exists")

    confirmation =  input("Press Enter to Run. ")

#pklデータの保存
with open(data_file_path, 'wb') as f:
    pickle.dump(data, f)

#入力と時間のグラフを作成する
plt.figure(0,figsize=(10, 5))
plt.scatter(time[k], steer_data[k], label='start')
#plt.ylim(-1.0, 0.2)
for j in range(len(steer_data)-1):
    i = j-1
    if not j == 0:
        if alpha[j] == 1:
            plt.plot([time[i], time[i+1]], [steer_data[i], steer_data[i+1]], label='assist on', color='red')
        elif alpha[j] == 0 and beta[j] == 0:
            plt.plot([time[i], time[i + 1]], [steer_data[i], steer_data[i + 1]], label='assist off', color='blue')
        else:
            plt.plot([time[i], time[i + 1]], [steer_data[i], steer_data[i + 1]], label='assist on (safe)', color='green')
plt.xlabel('time [s]')
plt.ylabel('input')
plt.xlim(60, 80)
plt.savefig(file_path_2)





#回避図
fig, ax = plt.subplots(1, figsize=(10, 5))

for i in range(len(ped_data)):
    plt.scatter(ped_data[i][0], ped_data[i][1], label='pedestrian')
plt.plot(data['car_x'], data['car_y'], label='car')

for point in ped_data:
    circle = Circle((point[0], point[1]), radius=2.0, edgecolor='red', facecolor='none',
                    label='pedestrian circle' if point == ped_data[0] else "")
    ax.add_patch(circle)

ax.set_aspect('equal', adjustable='datalim')
plt.xlabel(r'$y_1$')
plt.ylabel(r'$y_2$')
i = 0
plt.xlim(ped_data[i][0]-5.0, ped_data[i][0]+5.0)
plt.ylim(ped_data[i][1]-5.0, ped_data[i][1]+5.0)
# 凡例を追加
plt.legend()
plt.savefig(file_path)




# グラフを表示
plt.show()