from model import QNN
from params import *
import numpy as np


def train_DDQN(env):
    Q = QNN(env.history_t, 3, 64)
    step_max = env.train_df.shape[0]-1
    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    for epoch in range(epoch_num):
        curr_obs = env.reset()
        step = 0
        done = False
        total_loss = 0
        while not done and step < step_max:
            prev_obs = env.reset()
            step = 0
            done = False
            total_reward = 0
            total_loss = 0
            while not done and step < step_max:
                prev_obs_arr = np.reshape(prev_obs, (len(prev_obs), 1))
                # Select action
                if(np.random.random() < epsilon):
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(Q.forward(prev_obs_arr))

                # Take action
                curr_obs, reward, done = env.step(action)
                curr_obs_arr = np.reshape(curr_obs, (len(curr_obs), 1))
                # Store transition
                memory.append([prev_obs, action, reward, curr_obs, done])
                if(len(memory) > memory_size):
                    memory.pop(0)
                if(len(memory) == memory_size):
                    if(total_step % train_freq == 0):
                        shuffled_memory = np.random.permutation(memory)
                        memoery_index = range(len(memory))
