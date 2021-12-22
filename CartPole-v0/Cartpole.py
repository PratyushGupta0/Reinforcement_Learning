import gym
from Q_cart_pole import *
agent = Q_cart_pole()
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(np.append(env.reset(), np.array([0]), axis=0).shape)
curr_state = 0
new_state = 0
f = open("out.txt", "w")

for i_episode in range(11000):
    if(i_episode > 10000):
        agent.epsilon = 0.0
        print("Done")
    agent.lr = agent.lr * 0.999
    observation = env.reset()
    observation = np.reshape(observation, (4, 1))
    observation[observation > 5] = 5
    curr_state = observation
    curr_in = 0
    curr_target = 0
    for t in range(100):
        env.render()
        # print(observation.shape)
        action = agent.choose_action(curr_state)
       # print(action)
        observation, reward, done, info = env.step(action)
        observation = np.reshape(observation, (4, 1))
        if t == 0:
            curr_in = agent.preprocess(observation, action)
        else:
            curr_in = np.append(curr_in, agent.preprocess(
                observation, action), axis=0)
        new_state = observation
        if(t == 0):
            curr_target = agent.get_target_q(new_state, action, reward)
        else:
            curr_target = np.append(curr_target, agent.get_target_q(
                new_state, action, reward), axis=0)
        curr_state = new_state
        if done:
            f.write(str(i_episode)+"th iteration\n")
            f.write("Episode finished after {} timesteps\n".format(t+1))

            break
    agent.model.fit(curr_in, curr_target, epochs=50, verbose=1)
env.close()
f.close()
