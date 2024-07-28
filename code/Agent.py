class MineSweeper(nn.Module):
    def __init__(self, state_size, action_size, grid_size_X, grid_size_Y, environment):
        super(MineSweeper, self).__init__()
        self.render = False

        self.state_size = state_size
        self.action_size = action_size
        self.grid_size_X = grid_size_X
        self.grid_size_Y = grid_size_Y

        self.environment = environment

        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.target_update_counter = TARGET_UPDATE_COUNTER
        self.update_target_every = UPDATE_TARGET_EVERY

        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START
        self.maxlen = MAX_LEN
        self.minlen = MIN_LEN

        self.memory = deque(maxlen=self.maxlen)

        self.model = Net(self.action_size).to(device)
        self.target_model = Net(self.action_size).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=10000, mode='exp_range')

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        state = np.array(state).reshape(1, 1, self.grid_size_X, self.grid_size_Y)
        state = torch.FloatTensor(state).to(device)

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            self.q_value = q_value.detach().cpu().numpy().flatten()
            action = torch.argmax(q_value).item()

        return action

    def append_sample(self, state, action, reward, next_state, done):
        state = state
        next_state = next_state
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = states.reshape(self.batch_size, 1, self.grid_size_X, self.grid_size_Y)
        next_states = next_states.reshape(self.batch_size, 1, self.grid_size_X, self.grid_size_Y)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        pred = self.model(states)
        target_pred = self.target_model(next_states).max(1)[0].detach()

        targets = rewards + (1 - dones) * self.discount_factor * target_pred

        pred = pred.gather(1, actions.unsqueeze(1))
        trg = targets.unsqueeze(1)

        loss = self.loss(pred, trg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_update_counter = 0
            self.update_target_model()
