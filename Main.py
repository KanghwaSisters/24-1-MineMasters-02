device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###

DISCOUNT_FACTOR = 0.1
LEARNING_RATE = 0.01

EPSILON = 0.99
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

TARGET_UPDATE_COUNTER = 0
UPDATE_TARGET_EVERY = 5

BATCH_SIZE = 64
TRAIN_START = 1000
MAX_LEN = 50000

###

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

### Main ###

env = Environment()

state_size = env.state_size
action_size = env.state_size
grid_size_X = env.grid_size_X
grid_size_Y = env.grid_size_Y

agent = MineSweeper(state_size, action_size, grid_size_X, grid_size_Y, env)

EPISODES = 100000
RENDER_PROCESS = False
RENDER_END = False

total_moves = []
scores = np.zeros(EPISODES)
length_memory = np.zeros(EPISODES)
wins = np.zeros(EPISODES)
episodes = np.zeros(EPISODES)
timesteps = np.zeros(EPISODES)
win_rates = {}

N = 500
CHECKPOINT_INTERVAL = 10000

for epi in range(EPISODES):
    done = False
    score = 0
    time_step = 0
    actions = []
    rewards = []

    state = env.reset()

    last_loss = None

    while not done and time_step <= 82:
        time_step += 1
        if env.first_move:
            mine_state = env.minefield.flatten()
            first_action = random.randint(0, len(mine_state)-1)
            first_state = mine_state[first_action]
            while first_state == -1:
                first_action = random.randint(0, len(mine_state)-1)
                first_state = mine_state[first_action]
            action = first_action
            env.first_move = False
        else:
            action = agent.get_action(state)

        next_state, reward, done = env.step(action)
        score += reward

        (action_x, action_y) = divmod(action, env.grid_size_X)
        actions.append((action_x, action_y))
        rewards.append(reward)

        # state (신경망의 input) 정규화
        scaled_state = (next_state - (-1)) / (8 - (-1))

        agent.append_sample(state, action, reward, scaled_state, done)

        if len(agent.memory) >= agent.train_start:
            agent.train_model()

        state = next_state

    scores[epi] = score
    timesteps[epi] = time_step

    # 에피소드가 끝날 때 승리 여부를 기록
    if env.explode or time_step > 82:
        wins[epi] = 0
    elif not env.explode:
        wins[epi] = 1
        print(f"episode: {epi}")
        print(f"episode score: {score}")
        print(f"time step: {time_step}")
        print(f"epsilon: {agent.epsilon:.4f}")
        env.render()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if (epi+1) % N == 0:
        scores_N = np.median(scores[max(0, epi-N+1):epi+1])  # 마지막 N개의 요소에 대한 중간값 계산
        win_rate = np.mean(wins[max(0, epi-N+1):epi+1]) * 100  # 마지막 N개의 에피소드에 대한 승률 계산
        win_rates[epi] = win_rate
        length_memory[epi] = len(agent.memory)
        print(f"episode: {epi:3d} | time step: {time_step}")
        print(f"episode score: {score} | epsilon: {agent.epsilon:.4f}\n")
        print(f"<last {N} episode> score: {scores_N:.2f} | win rate: {win_rate:.2f}%\n")
        print(f"wins: {np.sum(wins[max(0, epi-N+1):epi+1])}\n")
        print(f"length of memory: {length_memory[epi]}\n")
        env.render()
        print(f"chosen_coordinate: {actions}")
        print(f"reward per time step: {rewards}")
        print("--------------------------------------------------")

    # 매 CHECKPOINT_INTERVAL마다 모델 저장
    if (epi+1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = f"checkpoint_{epi}.tar"
        save_checkpoint(agent, agent.optimizer, epi, checkpoint_path)
        print(f"Checkpoint saved at episode {epi} to {checkpoint_path}.")
