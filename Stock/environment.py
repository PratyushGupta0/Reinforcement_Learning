from preprocess import preprocess


class Environment():
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        self.train_df, self._ = preprocess()

    def reset(self):
        self.time = 0
        self.done = False
        self.profits = 0
        self.holdings = []
        self.holdings_value = 0
        self.history = [0 for i in range(0, self.history_t)]
        return [self.holdings_value]+self.history

    def step(self, action):
        reward = 0
        # Three actions: buy, sell, hold
        # {0,1,2} ->{buy,sell,hold}
        if action == 0:
            self.holdings.append(self.train_df.iloc[self.time])
        elif action == 1:
            if(len(self.holdings == 0)):
                reward = -1
            else:
                profits = 0
                for holding in self.holdings:
                    profits = profits + \
                        self.train_df.iloc[self.time:]['Close'] - holding
                reward = reward+profits
                self.holdings = []
        self.time = self.time+1
        self.postion_value = 0
        for holding in self.holdings:
            self.position_value = self.position_value + \
                self.train.iloc[self.time:]['Close']-holding
        self.history.pop(0)
        self.history.append(
            self.train_df.iloc[self.time:]['Close']-self.train_df.iloc[self.time-1:]['Close'])

        if(reward > 1.5):
            reward = 1.5
        if(reward < -1.5):
            reward = -1.5
        return [self.postion_value]+self.history, reward, self.done
