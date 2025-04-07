import matplotlib.pyplot as plt
import scipy
import numpy as np

import matplotlib.animation as animation

FS = 250
SUBJECT = 11

data = scipy.io.loadmat('STS_data_IvanVujaklija.mat', simplify_cells=True)
mocap_data = data['sts_data']['MOCAP'][SUBJECT]

N_SAMPLES = len(mocap_data)
DT = 1/FS
T_START = 0
T_END = N_SAMPLES * DT
t_data = np.linspace(T_START, T_END, N_SAMPLES, endpoint=False)

r_labels_index = np.array([0, 2, 3, 5, 7, 8, 9, 10])
l_labels_index = np.array([1, 2, 4, 6, 11, 12, 13, 14])

x_data = mocap_data[:, 3*r_labels_index+0]
y_data = mocap_data[:, 3*r_labels_index+1]
z_data = mocap_data[:, 3*r_labels_index+2]

class MyAnimation:
    def __init__(self, t_data, x_data, y_data, z_data, **kwargs):
        self.t_data = t_data
        self.x_data = x_data
        self.y_data = y_data
        self.z_data = z_data
        N = t_data.shape[0]

        fig = plt.figure()
        # front view
        ax_front = fig.add_subplot(121, autoscale_on=False,
                            xlim=(np.min(self.x_data)-0.1, np.max(self.x_data)+0.1),
                            ylim=(np.min(self.y_data)-0.1, np.max(self.y_data)+0.1))
        ax_front.set_aspect('equal')
        ax_front.grid()
        ax_front.set_xlabel('x')
        ax_front.set_ylabel('y')
        ax_front.set_title('Frontal Plane')

        self.dots_front = []
        for i in range(self.x_data.shape[1]):
            dot = ax_front.plot([], [], 'o')[0]
            self.dots_front.append(dot)

        # side view
        ax_side = fig.add_subplot(122, autoscale_on=False,
                            xlim=(np.min(self.z_data)-0.1, np.max(self.z_data)+0.5),
                            ylim=(np.min(self.y_data)-0.1, np.max(self.y_data)+0.1),
                            sharey=ax_front)
        ax_side.set_aspect('equal')
        ax_side.grid()
        ax_side.set_xlabel('z')
        ax_side.set_ylabel('y')
        ax_side.set_title('Sagittal Plane')

        self.dots_side = []
        for _ in range(self.z_data.shape[1]):
            dot = ax_side.plot([], [], 'o')[0]
            self.dots_side.append(dot)

        if 'labels' in kwargs:
            ax_side.legend(kwargs['labels'])
            
        self.ani = animation.FuncAnimation(
            fig, self.update,
            frames=N, interval=DT*1000, blit=True)

        self.paused = False
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, event):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def update(self, i):
        for j in range(self.x_data.shape[1]):
            self.dots_front[j].set_data([self.x_data[i, j]], [self.y_data[i, j]])
            self.dots_side[j].set_data([self.z_data[i, j]], [self.y_data[i, j]])
        return self.dots_front + self.dots_side

ani = MyAnimation(t_data, x_data, y_data, z_data, labels=data['mocap_labels'][r_labels_index])
plt.show()