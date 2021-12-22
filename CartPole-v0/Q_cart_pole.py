import tensorflow as tf
import numpy as np


class Q_cart_pole:
    def __init__(self, learning_rate=0.01, reward_decay=1, e_greedy=0.9):
        self.n_actions = 2
        self.n_features = 4
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(
                units=256, activation='relu'),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=4, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu'),

        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.lr), loss='mse')

    def preprocess(self, observation, action):
        return np.append(np.reshape(observation, (1, observation.shape[0])), np.reshape(action, (1, 1)), axis=1)

    def choose_action(self, observation):
        curr_state = observation
        sample = np.random.choice([0, 1], p=[self.epsilon, 1-self.epsilon])
        if(sample == 0):
            action = np.random.randint(0, self.n_actions)
        else:
            pred_zero = self.model.predict(self.preprocess(curr_state, 0))
            pred_one = self.model.predict(self.preprocess(curr_state, 1))
            action = np.argmax(np.array([pred_zero, pred_one]))  # 0 or 1
        return action

    def get_target_q(self, observation, action, reward):
        return np.reshape(self.model.predict(self.preprocess(observation, action))[0][0]*self.gamma+reward, (1, 1))

    def learn(self, curr_state, curr_action, reward, new_state):
        new_action = self.choose_action(new_state)
        q_target = self.get_target_q(new_state, new_action, reward)
        self.model.fit(self.preprocess(curr_state, curr_action),
                       q_target, epochs=5, verbose=0)
