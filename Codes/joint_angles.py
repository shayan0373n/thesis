import matplotlib.pyplot as plt
import scipy
import numpy as np

import matplotlib.animation as animation

FS = 240
SUBJECT = 10
reversed = False # whether the subject is facing the z-axis or is reversed

data = scipy.io.loadmat('STS_data_IvanVujaklija.mat', simplify_cells=True)
labels = list(data['mocap_labels'])
mocap_data_raw = data['sts_data']['MOCAP'][SUBJECT]

N_SAMPLES = len(mocap_data_raw)
DT = 1/FS
T_START = 0
T_END = N_SAMPLES * DT
t = np.linspace(T_START, T_END, N_SAMPLES, endpoint=False)

mocap_data = {}
for label in labels:
    i = labels.index(label)
    # 2D projection in the sagittal plane
    mocap_data[label] = mocap_data_raw[:, 3*i+1:3*i+3]
    if reversed:
        # reverse z-axis
        mocap_data[label][:, 1] = -mocap_data[label][:, 1]

foot_r = mocap_data['met_r'] - mocap_data['mal_r']
# rotate by 90 degrees around x-axis
foot_r = np.column_stack((foot_r[:,1], -foot_r[:,0]))
lleg_r = mocap_data['ph_r'] - mocap_data['mal_r']
uleg_r = mocap_data['gt_r'] - mocap_data['fh_r']
pelvis = 0.5*(mocap_data['iliac_r'] + mocap_data['iliac_l']) - mocap_data['sacrum']
# rotate by 90 degrees around x-axis
pelvis = np.column_stack((pelvis[:,1], -pelvis[:,0]))
torso = 0.5*(mocap_data['sh_r'] + mocap_data['sh_l']) - mocap_data['sacrum']

# calculate joint angles
hip_r = np.arctan2(uleg_r[:, 0], uleg_r[:, 1]) - np.arctan2(pelvis[:, 0], pelvis[:, 1])
knee_r = np.arctan2(lleg_r[:, 0], lleg_r[:, 1]) - np.arctan2(uleg_r[:, 0], uleg_r[:, 1])
ankle_r = np.arctan2(foot_r[:, 0], foot_r[:, 1]) - np.arctan2(lleg_r[:, 0], lleg_r[:, 1])
torso_angle = np.arctan2(torso[:, 0], torso[:, 1]) - np.arctan2(pelvis[:, 0], pelvis[:, 1])

# plot joint angles
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, hip_r*180/np.pi)
axs[0, 0].set_title('Hip')
axs[0, 0].set_ylabel('Angle (deg)')
axs[0, 1].plot(t, knee_r*180/np.pi)
axs[0, 1].set_title('Knee')
axs[1, 0].plot(t, ankle_r*180/np.pi)
axs[1, 0].set_title('Ankle')
axs[1, 0].set_ylabel('Angle (deg)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 1].plot(t, torso_angle*180/np.pi)
axs[1, 1].set_title('Torso')
axs[1, 1].set_xlabel('Time (s)')
for ax in axs.flatten():
    ax.grid()
    ax.set_xlim([0, t[-1]])
plt.show()