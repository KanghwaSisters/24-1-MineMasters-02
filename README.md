# 24-1-MineMasters-02
[ 24-1 /  MineMasters / Team 02 ]  
👩‍💻 이정연, 손주현
***
# 목차
[Environment](#Environment)

[Net](#Net)

[Agent](#Agent)

[Train](#Train)

[코드 속도 개선](#코드-속도-개선)

[시도한 방법론, 아이디어 결과 및 분석](#시도한-방법론-아이디어-결과-및-분석)

[최고 성능이 나온 모델](#최고-성능이-나온-모델)

[문제 해결 및 개선한 점](#문제-해결-및-개선한-점)

***

 # Environment

## Attributes

- **`grid_size_X`** , **`grid_size_Y`** : 게임판의 가로, 세로 크기
- **`num_mines`** : 지뢰 개수
- **state**
    - **`state_size`**: 지뢰판 크기
    - **`minefield`**  : 지뢰 찾기 게임 상황이 저장된 애트리뷰트(정답지)
        
        - **type:** np.array
        - **구조:** `grid_size_X` x `grid_size_Y` 크기의 2D NumPy 배열
        - 초기값: 모든 셀이 0 → 이후 `reset()` 내 `place_mines()`에서 지뢰를 심는다. (-1로 update)
    - **`playerfield`** : 플레이어가 보는 지뢰밭 상태
        
        *-1(mine), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9(hidden)* 
        
        0~8은 인접한 지뢰 개수를 나타낸다.
        
        - **type:** np.array
        - **구조:** `grid_size_X` x `grid_size_Y` 크기의 2D NumPy 배열
        - **초기값:** 모든 셀이 9
- **게임 진행 관련 변수들**
    - **`explode`**: 플레이어가 지뢰를 밟았는지 여부
    - **`done`**  : 게임이 끝났는지 여부
    - **`first_move`**: 현재 움직임이 첫 번째 움직임인지 여부
- 이때 **`action`** 은 에이전트가 playerfield에서 선택한 좌표로 정의한다.
- 보상 **`rewards`**
    | explode | noprogress | guess | progress | clear |
    | --- | --- | --- | --- | --- |
    | -1 | -1 | 0.1 | 0.3 | 1 |
    - 'guess' : 0.1
        - 주변을 둘러싼 타일들이 전부 hidden tile인 경우 (아무런 정보 없이 찍었다고 간주함)
    - 'progress' : 0.3
        - 주변을 둘러싼 타일 중에 하나라도 open된 타일이 있는 경우
        - 가장자리 타일은 주변을 둘러싼 타일이 5개 / 꼭짓점 타일은 3개

## Methods

### 1. **`__init__`**
게임판 크기, minefield 크기, playerfield 크기, state 크기, 폭발 여부, done 여부, 첫 스텝 여부, 방문 좌표 set, reward 배열 등의 변수를 초기화한다.

```python
class Environment:
    def __init__(self):
        self.grid_size_X = 9
        self.grid_size_Y = 9
        self.num_mines = 10

        self.minefield = np.zeros((self.grid_size_X, self.grid_size_Y), dtype=int)

        self.playerfield = np.full((self.grid_size_X, self.grid_size_Y), 9, dtype=int)

        self.state_size = self.minefield.size

        self.explode = False
        self.done = False
        self.first_move = True
        self.visited = set()

        self.rewards = {'explode' : -1, 'noprogress' : -0.1,'progress' : 0.3, 'guess' : 0.1, 'clear' : 1}
```

### 2. **`reset`**
에피소드를 초기 상태로 되돌리는 역할
새로운 게임에 필요한 지뢰를 배치하고, 새로운 playerfield 를 제공한다.
- `place_mines`  : 지뢰 개수만큼 임의의 좌표에 지뢰를 심은 후, 지뢰가 없는 좌표에 대해서는 인접한 지뢰개수를 playerfield에 update한다.
- `count_adjacent_mines` : 인접한 지뢰 개수를 세는 메소드

```python
    def reset(self):
        self.minefield = np.zeros((self.grid_size_X, self.grid_size_Y), dtype=int)
        self.playerfield = np.full((self.grid_size_X, self.grid_size_Y), 9, dtype=int)

        self.explode = False
        self.done = False
        self.first_move = True

        self.visited = set()

        self.place_mines()

        return list(self.playerfield)
```
```python
    def place_mines(self):
        mines_placed = 0

        # num_mines만큼 임의의 좌표에 지뢰 심기
        while mines_placed < self.num_mines:
            x = random.randint(0, self.grid_size_X - 1)
            y = random.randint(0, self.grid_size_Y - 1)

            if self.minefield[x, y] == 0:
                self.minefield[x, y] = -1
                mines_placed += 1

        # 지뢰 없는 좌표: 인접 지뢰 개수 세기
        for x in range(self.grid_size_X):
            for y in range(self.grid_size_Y):
                if self.minefield[x, y] == -1:
                    continue
                self.minefield[x, y] = self.count_adjacent_mines(x, y)

    def count_adjacent_mines(self, x, y):
        count = 0
        # (x,y) 주변 지뢰 개수
        for i in range(max(0, x - 1), min(self.grid_size_X, x + 2)):
            for j in range(max(0, y - 1), min(self.grid_size_Y, y + 2)):
                if (i, j) != (x, y) and self.minefield[i, j] == -1:
                    count += 1
        return count
```

### 3. **`step`**
에이전트가 환경에서 action을 한 단계 수행할 때마다 호출

- `next_state`, `reward`, `done` 반환
- 동작 과정
    1. 1차원 인덱스 `action` 를 2차원 좌표로 변환한다.
        
        (minefield, playerfield 모두 2차원이기 때문에)
        
    2. 선택한 좌표 (x,y)가 지뢰인지 여부를 minefield에서 확인
        1. 지뢰(-1)인 경우
            - `done`, `explode`, `reward` 로 각각 True, True, rewards['explode'] 반환
            - 게임 패배
        2. 지뢰(-1)가 아닌 경우
            1. 선택한 좌표가 이미 open된 좌표인 경우
                - `done`, `explode`, `reward` 로 각각 False, False, rewards['noprogress'] 반환
                - `next_state`로 `playerfield` 반환
            2. 선택한 좌표가 처음 open된 좌표인 경우
                1. `visited` 배열에 타일 추가
                2. 타일 open (playerfield의 좌표에 minefield의 해당 좌표 값을 복사)
                    - 중심부 타일은 주변을 둘러싼 타일이 8개 / 가장자리 타일은 5개 / 꼭짓점 타일은 3개
                        1. open한 타일이 주변이 전부 hidden인 경우
                            - `reward`로 rewards['guess'] 부여
                        2. open한 타일 주변에 이미 open된 타일이 있는 경우
                            - `reward`로 rewards['progress'] 부여
                3. open한 타일이 0이면 주위 타일 open (`auto_reveal_tiles`)
                4. hidden tile(9)이 남아있는 경우 `done=False`, 남아있지 않은 경우 `done=True` 반환
                5. `next_state`로 playerfield 반환
        ![step()_flowchart](https://github.com/user-attachments/assets/c80f349a-d6cb-4ca9-9df2-f28cd16763ef)

```python
def step(self, action):
        x, y = divmod(action, self.grid_size_X)

        reward = 0
        done = False

        # explode: 지뢰 선택 시 done
        if self.minefield[x, y] == -1:
            self.playerfield[x, y] = self.minefield[x, y]  # 타일 열기
            self.explode = True
            done = True
            reward = self.rewards['explode']

        # 지뢰를 선택하지 않은 경우
        else:
          # noprogress: 선택한 좌표 (x,y)가 이미 방문된 경우
            if (x, y) in self.visited:
                reward = self.rewards['noprogress']
          # 선택한 좌표 (x, y)가 처음 방문된 경우
            else:
                self.playerfield[x, y] = self.minefield[x, y]  # 타일 열기
                self.visited.add((x,y))
                # 가장자리 타일
                if x in [0, 8] or y in [0, 8]:
                    # guess
                    if self.count_adjacent_hidden(x, y) == 5:
                        reward = self.rewards['guess']
                    # progress
                    else:
                        reward = self.rewards['progress']
                # 꼭짓점 타일
                elif x in [0, 8] and y in [0, 8]:
                    # guess
                    if self.count_adjacent_hidden(x, y) == 3:
                        reward = self.rewards['guess']
                    # progress
                    else:
                        reward = self.rewards['progress']
                else:
                    if self.count_adjacent_hidden(x, y) == 8:
                        reward = self.rewards['guess']
                    # progress
                    else:
                        reward = self.rewards['progress']
                # open한 타일이 0이면 주위 타일 open
                if self.playerfield[x, y] == 0:
                  self.auto_reveal_tiles(x, y)  # (x, y) 주변 타일 열기

            # clear: 모든 hidden 타일이 지뢰만 남아 있는 경우 승리
            if np.count_nonzero(self.playerfield == 9) == self.num_mines:
                done = True
                reward = self.rewards['clear']

        self.done = done
        next_state = self.playerfield
        return next_state, reward, done
```
- 타일을 open할 때 필요한 메서드
    - `check_boundary` : open할 타일이 게임판을 벗어나지 않도록
    - `auto_reveal_tiles`  : 0을 선택한 경우 연쇄적으로 주위 타일을 전부 open
        - BFS 구조
        - 주변 8개 타일의 숫자 확인, 게임판을 벗어나지 않고 방문하지 않은 경우 큐에 추가
     
- 선택한 타일 주변 hidden tile 개수를 세는 데에 필요한 메서드
    - `count_adjacent_hidden`

```python
    def check_boundary(self, x, y):
        return 0 <= x < self.grid_size_X and 0 <= y < self.grid_size_Y

    def auto_reveal_tiles(self, x, y):  # BFS
        queue = deque([(x, y)])

        while queue:
            cx, cy = queue.popleft()
            self.visited.add((cx, cy))  # (cx, cy) 방문 표시
            self.playerfield[cx, cy] = self.minefield[cx, cy]  # (cx, cy) 타일 열기

            # (cx, cy) 주변 8개 타일 확인, 범위 내에 있으면 큐에 insert
            if self.minefield[cx, cy] == 0: # 방문하지 않았으면 open
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        # 인덱스가 게임판 범위 내에 있는지 확인
                        if self.check_boundary(nx, ny) and (nx, ny) not in self.visited and (nx, ny) not in queue:  # nonvisited 주위 타일 큐에 추가
                            queue.append((nx, ny))
```
```python
    def count_adjacent_hidden(self, x, y):
        count = 0
        # (x,y) 주변 hidden tile 개수
        for i in range(max(0, x - 1), min(self.grid_size_X, x + 2)):
            for j in range(max(0, y - 1), min(self.grid_size_Y, y + 2)):
                if (i, j) != (x, y) and self.playerfield[i, j] == 9:
                    count += 1
        return count
```

### 4. **`render`** 
특정 시점에 playerfield 게임판의 상태를 render
- Hidden tile: **.**
- Mine: **X**
- 나머지: **0~8** (인접한 지뢰 수)

```python
def render(self):  # 인수 설정
        for x in range(self.grid_size_X):
            for y in range(self.grid_size_Y):
                tile = self.playerfield[x, y]
                if tile == 9:
                    print('.', end=' ')
                elif tile == -1:
                    print('X', end=' ')
                else:
                    print(tile, end=' ')
                if y == self.grid_size_Y - 1:
                    print()
        print('\n')
```

***
# Net
```python
class Net(nn.Module):
    def __init__(self, action_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, action_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        return x
```

- 배치 정규화 (Batch normalization)
    - 기대할 수 있는 장점
        - 학습 속도가 빨라짐
        - 파라미터 초기화에 덜 민감함
        - 모델을 일반화
- 풀링 (Pooling)
    - Pooling layer는 convolution layer의 출력 데이터를 입력으로 받는다. 그리고 출력 데이터의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용된다.
    - Net의 파라미터 개수나 연산량을 줄이기 위해 downsampling하는 것이다.
    - Receptive field를 크게 만들어 전역적 특징을 더 잘 포착하도록 할 수 있다.

***
# Agent

### Hyperparameters

```python
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
```

- **DISCOUNT_FACTOR**: 미래 보상의 현재 가치에 대한 할인 계수 (Gamma)
- **LEARNING_RATE**: Optimizer에 적용할 학습률
- **EPSILON**: 탐험을 할 확률을 조절하는 값 (초기 0.99)
- **EPSILON_MIN**: (최소 0.01)
- **EPSILON_DECAY** 매 에피소드가 끝날 때마다 입실론에 곱하여 입실론 값을 작아지게 하는 값
- **BATCH_SIZE**: 리플레이 메모리에서 샘플링할 배치 크기
- **TRAIN_START**: 학습을 시작할 리플레이 메모리의 최소 크기
- **MAX_LEN**: 리플레이 메모리의 최대 길이
- **TARGET_UPDATE_EVERY**: 타깃 네트워크의 가중치를 업데이트할 주기

### Methods

1. **`__init__`**

상태 크기, 행동 크기, 하이퍼파라미터, 리플레이 메모리, 모델, 타깃 모델 등을 초기화한다.

```python
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
```
2. **`update_target_model`**
    
    타깃 모델을 모델의 가중치로 업데이트한다.
    

```python
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
```

3. **`get_action`**
    
    입실론 탐욕 정책을 사용하여 행동을 결정한다. 탐험을 통해 무작위로 행동을 결정하거나, 인공 신경망을 통해 탐욕적으로 행동을 선택한다.
    
    - `action = torch.argmax(q_value).item()`
        
        반환값 : `q_value` 텐서에서 최대값을 가지는 요소 (여러 개여도 첫 번째 하나의 요소(`torch.argmax()`)의 인덱스를 나타내는 정수 값(`.item()`)
        

```python
    def get_action(self, state):
        state = np.array(state).reshape(1, 1, self.grid_size_X, self.grid_size_Y)
        state = torch.FloatTensor(state).to(device)

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            action = torch.argmax(q_value).item()

        return action
```

4. **`append_sample`**
    
    샘플을 리플레이 메모리에 저장한다.
    

```python
    def append_sample(self, state, action, reward, next_state, done):
        state = state
        next_state = next_state
        self.memory.append((state, action, reward, next_state, done))
```

5. **`train_model`**
    
    리플레이 메모리에서 샘플링한 배치로 모델을 학습하고 타깃 모델을 주기적으로 업데이트한다.
    
    - **모델 학습 과정**
        - 리플레이 메모리에서 무작위로 샘플링하여 배치 생성
        - 현재 모델을 사용하여 상태의 Q-value 예측
        - 타깃 신경망을 사용하여 다음 상태의 최대 Q-value을 계산하고 벨만 최적 방정식으로 타깃 업데이트
        - 예측한 Q값과 타깃 Q값의 차이를 계산하여 loss 구함
        - loss를 기반으로 모델 가중치 업데이트 및 학습률 조정

```python
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
```

***
# Train

- **멈춤 조건 설정**: 지뢰찾기 게임에서 승리하려면 81개의 좌표 중 지뢰 10개를  제외하고 나머지는 전부 연 상태가 되어야 하므로 time step이 71 (81-10)에 도달하면 에피소드가 종료되도록 했다.
- **state 정규화**: 신경망의 입력이 되는 state를 $value - min\over {max - min}$ 공식을 이용해 정규화한 후 리플레이 메모리에 샘플로 저장하였다.
- **활용 지표**
    - Episode score: time step 당 얻는 reward의 합
    - Last N episode score: 학습을 진행하며 마지막 N(=500)개의 에피소드들의 score의 중간값
    - wins: 에피소드의 승리 여부를 저장한 배열
    - win rate: N개의 에피소드 중 승리한 비율
    - 승리 여부 확인: 에이전트가 에피소드 종료까지 71 time step 이내에 어떤 지뢰도 선택하지 않고 나머지 타일은 전부 열었다면, 넘파이 배열 wins에 1을, 그렇지 않으면 0을 넣는다. 마지막 500개의 에피소드들에 해당하는 wins의 요소 중 1의 개수를 승리한 횟수로 하여 승리 횟수 및 승률을 확인하였다.
- CheckPoint마다 모델을 저장했다.
  
***
# 코드 속도 개선

1. 리스트와 넘파이 배열
    
    `UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:274.)
    states = torch.tensor(states, dtype=torch.float32).to(device)` 
    
    ```python
    # Agent 클래스의 train_model() 메서드
    
    minibatch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    # 여기서 각 요소들 불러오면 리스트로 받아지므로
    
    # 넘파이 배열들의 리스트 -> 다차원 넘파이 배열로
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    ```
    
2. BFS
    - **`auto_reveal_tiles` 메서드 구현 문제**
        - 재귀 호출 방식 대신 반복문과 큐를 이용하는 방식으로 변경했다.
        - 기존 코드에서는 재귀 호출 방식 사용 결과, 너무 많은 시간이 소요되는 문제 발생
        
            ```python
            def auto_reveal_tiles(self, x, y):
                    visited = set()  # 중복된 값 허용 X
                    
                    def reveal(x, y):
                        if (x, y) in visited:
                            return
                        visited.add((x, y))
                        self.playerfield[x, y] = self.minefield[x, y]
            
                        # 주변 8개 타일 확인
                        if self.minefield[x, y] == 0:
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = x + dx, y + dy
                                    # 인덱스가 게임판 범위 내에 있는지 확인
                                    if self.check_boundary(nx, ny) and (nx, ny) not in visited:
                                        reveal(nx, ny)
                    reveal(x, y)
                    return self.playerfield
            ```
        
        - 큐와 반복문을 이용한 BFS(너비우선탐색) 구조로 변경 ⇒ 속도 개선
            
            ```python
                def auto_reveal_tiles(self, x, y):  # BFS
                    queue = deque([(x, y)])
                    self.visited = set()
            
                    while queue:
                        cx, cy = queue.popleft()
                        self.visited.add((cx, cy))  # (cx, cy) 방문 표시
                        self.playerfield[cx, cy] = self.minefield[cx, cy]  # (cx, cy) 타일 열기
                        self.visit_count[(cx, cy)] = self.visit_count.get((cx, cy), 0) + 1  # 방문 횟수 기록
            
                        # (cx, cy) 주변 8개 타일 확인, 범위 내에 있으면 큐에 insert
                        if self.minefield[cx, cy] == 0: 
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = cx + dx, cy + dy
                                    # 인덱스가 게임판 범위 내에 있는지 확인
                                    if self.check_boundary(nx, ny) and (nx, ny) not in self.visited and (nx, ny) not in queue:  # nonvisited 주위 타일 큐에 추가
                                        queue.append((nx, ny))
            ```
            
***
# 시도한 방법론, 아이디어 결과 및 분석

1. 먼저 **게임판을 10개로 한정**하여 성능(승률)을 높이는 것을 시도함
    
    단순한 구조의 DNN을 net으로 사용하고 learning rate scheduler가 lambdaLR일 때는  학습 시 거의 한 번도 승리하지 못하다가, CNN과 cyclicLR으로 변경하니 50000 에피소드로 학습 시 평균 승률 3.6%를 웃돌았다.
    
    ![10-highest_win_rate](https://github.com/user-attachments/assets/449a6fd3-38a6-48f5-8b45-48cb35f0f523)
    
    ![10-timesteps](https://github.com/user-attachments/assets/60eab1f8-aa81-4cff-967e-530639f6b268)
   
    ![10-score](https://github.com/user-attachments/assets/22b6c5eb-fdb3-468b-8306-d7463eb8d704)

   (rewards = {'explode' : -20, 'open_nonzero' : 5, 'open_zero' : 10, 'clear' : 20})
   
3. **state 정규화**
    
    Net의 input으로 이용할 state를 정규화하였다.
    
    - **정규화의 목적** : 그래디언트 소실 및 폭파 문제가 발생하지 않도록 input의 절댓값을 특정 범위로 제한하기 때문에 그래디언트가 saturate되지 않도록 하여 더 빠른 최적화가 가능해진다.
    - **state 정규화를 시행한 위치**
        
        정규화를 step에서 하면 다음 step의 state가 scaled_state가 되는 문제 발생
        
        (state과 scaled_state 간의 구분 불가)
        
        ⇒ env.step() 대신 학습 루프에 scaled_state를 정의
        
        ```python
        for epi in range(EPISODES):
        		...
            state = env.reset() # 2차원 배열 state
        
            while not done and time_step <= 71:
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
        
                # state (신경망의 input) 정규화
                scaled_state = (next_state - (-1)) / (8 - (-1))
        
                agent.append_sample(state, action, reward, scaled_state, done)
        ```
        
4. **validation - 과적합 문제**
    
    train에 비해 test의 성능이 현저하게 떨어지는 문제 발생
    
    ![valid-train](https://github.com/user-attachments/assets/357af465-3b75-4941-8c6f-2e50628fb58f)
    Train
    
    ![valid-test](https://github.com/user-attachments/assets/e56a3a9f-a19d-4899-a713-9e950864f4a6)
    Test
    
    ⇒ validation 코드를 추가하여 과적합을 방지하고자 하였다.
    
    - validation 환경에서 에이전트를 평가하고 평균 점수와 승률을 출력한다.
    - 학습 중간중간 에이전트의 성능을 체크하며, 최적의 하이퍼파라미터를 선택하거나 모델이 과적합되지 않도록 방지하는 데 도움이 되는 Validation을 추가했다.
    - 학습 시 초기 에피소드들에서는 탐험을 통해 학습이 빠르게 변할 수 있으므로 더 자주 Validation을 하는 것이 좋을 것 같다고 생각해서 초기의 Validation 주기를 1000으로 설정했다.
    - Validation
        - 20000번째 에피소드 전까지는 1000을 주기로,
        - 60000번째 에피소드 전까지는 5000을 주기로,
        - 이후에는 20000을 주기로 validation
    
    ```python
    class ValidationEnvironment(Environment):
        def __init__(self):
            super().__init__()
    ```
    
    ```python
    class MineSweeper(nn.Module):
    		...
    		def validate_model(self, validation_env, episodes=100):
    	      self.model.eval()
            total_score = 0
            total_wins = 0
    
            for epi in range(episodes):
                state = validation_env.reset()
                done = False
                score = 0
    
                while not done:
                    action = self.get_action(state)
                    next_state, reward, done = validation_env.step(action)
                    score += reward
                    state = next_state
    
                total_score += score
                if not validation_env.explode:
                    total_wins += 1
    
            avg_score = total_score / episodes
            win_rate = (total_wins / episodes) * 100
    
            print(f"Validation results over {episodes} episodes:")
            print(f"Average score: {avg_score:.2f}")
            print(f"Win rate: {win_rate:.2f}%")
    
            self.model.train()
    ```
    
    ```python
    VALIDATION_INTERVAL_INITIAL = 1000
    VALIDATION_INTERVAL_MIDDLE = 5000
    VALIDATION_INTERVAL_LATE = 20000
    
    validation_env = ValidationEnvironment()
    
    ...
        
        if epi < 20000:
            validation_interval = VALIDATION_INTERVAL_INITIAL
        elif epi < 60000:
            validation_interval = VALIDATION_INTERVAL_MIDDLE
        else:
            validation_interval = VALIDATION_INTERVAL_LATE
    
        if epi % validation_interval == 0:
            print(f"Performing validation at episode {epi}...")
            agent.validate_model(validation_env, episodes=100)
            print("Validation complete.")
    ```
    
    - 10만 번~20만 번의 에피소드로 모델을 학습시키면서 최대한 비용(Cost)를 절감하기 위해 최종 모델에서는 validation을 사용하지 않았다.
5. 에이전트가 **같은 타일을 선택**할 수 있게 할 것인가?
    - 한 에피소드 내에서 한 번 선택한 타일은 이후의 time step에서 선택하지 못하도록 하는 방법: 오히려 에이전트가 ‘이미 연 타일을 반복해서 선택하는 것은 좋지 않은 행동’임을 학습하는 걸 방해하는 것이라고 생각하게 되었다.
    - 하지만 같은 타일을 계속 선택하다 보면 time step이 엄청나게 길어질 수 있다.
        
        → 최대 time step 71(81-10)으로 제한했다. (멈춤 조건 설정)
        
6. 지뢰를 선택했을 때와 방문했던 좌표를 또 선택할 때의 보상이 동일함에도 불구하고, 학습이 진행됨에 따라 지뢰는 거의 선택하지 않는데 이미 갔던 (안전하다고 판단하는) 좌표는 계속 방문하는 문제 발생: 
    
    이미 방문한 타일을 다시 방문할 때마다 페널티를 점진적으로 증가시키는 방법을 사용하여 에이전트가 동일한 타일을 반복해서 방문하지 않도록 시도해보았다.
    
    ```python
    self.visit_count = {}
    
    self.rewards = {'explode' : -1, 'nonprogress' : -1,'open_nonzero' : 0.1, 'open_zero' : 0.3, 'clear' : 1}
    
    if (x, y) in self.visit_count:  # 선택한 좌표 (x,y)가 이미 방문된 경우
    		self.visit_count[(x, y)] += 1  # 방문 횟수 증가
    		reward = self.rewards['nonprogress'] * self.visit_count[(x, y)]
    ```
    
    방문 타일 보상 페널티 누적 여부에 따라 성능이 크게 달라지지 않았다.
    
7. 보상 설정
    1. 지뢰 선택 시 (---) / 지뢰 아닌 좌표 선택 시 (+) / 승리 시 (++)
    2. 지뢰 선택 시 (---) / 0을 선택하여 많은 좌표가 열렸을 시 (++) / 0이 아닌 숫자 좌표 선택 시 (+) / 승리 시 (+++)
    3. 지뢰 선택 시 (---) / 이미 연 좌표 선택 시 (---) / 새로운 0 선택 시 (++) / 새로운 0이 아닌 숫자 좌표 선택 시 (+) / 승리 시 (+++)
    
    최종 모델에는 c.의 보상 설정 방식을 선정했다.
    
8. Optimizer: Adam과 RMSprop
9. Net: DNN과 CNN / 기본적인 CNN과 ResNet을 참고한 구조
    
    ```python
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
    
        def forward(self, x):
            residual = self.downsample(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            return out
    ```
    
    ![residual_learning](https://github.com/user-attachments/assets/8b835ecd-78a6-4e56-9b45-92c0a2f6c017)
    
    - 2015년에 개최된 ILSVRC(ImageNet Large Scale Visual Recognition Challenge)에서 우승을 차지하고 딥러닝 이미지 분야에서 많이 사용되고 있는 ResNet의 구조를 참고했다.
    - ResNet은 Residual Learning을 이용하는데, 위 그림의 $F(x)$  (잔차) + $x$ 를 최소화하는 것을 목적으로 한다. ResidualBlock 클래스의 forward 메서드에서 이 구조를 따랐다.
    
    ```python
    class Net(nn.Module):
        def __init__(self, grid_size_X, grid_size_Y, action_size):
            super(Net, self).__init__()
            self.in_channels = 64
    
            self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
            self.bn = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=1)
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, action_size)
    
        def _make_layer(self, out_channels, blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)
    
        def forward(self, x):
            out = F.relu(self.bn(self.conv(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            # 텐서의 첫 번째 (인덱스가 0) 차원 -배치 크기- 은 고정하고 나머지 차원의 크기를 곱해 2차원 텐서로 변환
            out = self.fc(out)
            return out
    ```
    
    Residual Learning을 이용한 Net을 사용한 경우, 모델이 무거워지고 그렇다고 해서 성능이 크게 향상되지도 않았기 때문에 최종 모델에는 사용하지 않았다.
    
10. Learning rate scheduler: lambdaLR / cyclicLR / StepLR
11. 모델 저장(추론 / 학습 재개를 위해 일반 체크포인트(checkpoint) 저장하기 & 불러오기**)**
    - 체크포인트 저장하기
        
        ```python
        if epi % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"checkpoint_{epi}.tar"
            save_checkpoint(agent, agent.optimizer, epi, score, checkpoint_path)
            print(f"Checkpoint saved at episode {epi} to {checkpoint_path}.")
        ```
        
    - 체크포인트 불러오기
        
        ```python
        checkpoint_path = 'checkpoint_5000.tar'  # 예) 5000번째 에피소드 체크포인트
        agent, optimizer, start_epoch, last_loss = load_checkpoint(agent, optimizer, checkpoint_path)
        
        print(f"Checkpoint loaded from {checkpoint_path}. Starting from epoch {start_epoch}.")
        ```
***
# 최고 성능이 나온 모델

- **Adam optimizer를 이용한 모델**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/1c873709-ed4b-4a75-ae8b-055a2c375a93/b665a0c8-4368-46fb-9d5c-ce4ea58fca86/Untitled.png)

- **RMSprop optimizer를 이용한 모델**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/1c873709-ed4b-4a75-ae8b-055a2c375a93/f457119c-a946-4bf8-a444-5fac91cb0cb0/Untitled.png)

***
# 문제 해결 및 개선한 점

- Train의 #safe first click 부분 수정
    - 기존: 첫 좌표부터 돌면서 -1이 아닌 처음 좌표 선택 → 처음 open하는 좌표가 항상 비슷해짐 (왼쪽 상단 부분에 위치한 좌표들만 선택하게 됨)
    - 수정: 전체 좌표에서 랜덤 선택 → 지뢰 선택했으면 다시 다른 좌표 랜덤 선택
    
    ```python
    while not done and time_step <= 71:
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
    ```
    
- **score 산정 방식**
    - step 수가 많을수록 비효율적이라는 점 고려
    
    각 에피소드의 전체 보상들의 median으로 설정했더니 대부분 동일한 값이 나오던 문제
    
    ⇒ 각 에피소드의 score을 해당 에피소드의 보상 총합으로 수정
    
- **가능하면 상수에 대해 하이퍼파라미터 설정**
    
     ⇒  코드의 독립성 향상
    
- **탐험 부족**
    - epsilon_decay 값을 늘림→ epsilon이 더 천천히 감소하도록
    - epsilon_min 값을 줄여 나중에는 정책에 더 의존할 수 있도록
    - 여러 배치 크기 시도
