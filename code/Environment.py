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

    def reset(self):
        self.minefield = np.zeros((self.grid_size_X, self.grid_size_Y), dtype=int)
        self.playerfield = np.full((self.grid_size_X, self.grid_size_Y), 9, dtype=int)

        self.explode = False
        self.done = False
        self.first_move = True

        self.visited = set()

        self.place_mines()

        return list(self.playerfield)

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

    def count_adjacent_hidden(self, x, y):
        count = 0
        # (x,y) 주변 hidden tile 개수
        for i in range(max(0, x - 1), min(self.grid_size_X, x + 2)):
            for j in range(max(0, y - 1), min(self.grid_size_Y, y + 2)):
                if (i, j) != (x, y) and self.playerfield[i, j] == 9:
                    count += 1
        return count

    def step(self, action):
        x, y = divmod(action, self.grid_size_X)

        reward = 0
        done = False

        # explode: 지뢰 선택 시 done
        if self.minefield[x, y] == -1:
            self.playerfield[x, y] = self.minefield[x, y]
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
                self.playerfield[x, y] = self.minefield[x, y]
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
                # 중심부 타일
                else:
                    if self.count_adjacent_hidden(x, y) == 8:
                        reward = self.rewards['guess']
                    # progress
                    else:
                        reward = self.rewards['progress']
                # open한 타일이 0이면 주위 타일 open
                if self.playerfield[x, y] == 0:
                  self.auto_reveal_tiles(x, y)

            # clear: 모든 hidden 타일이 지뢰만 남아 있는 경우 승리
            if np.count_nonzero(self.playerfield == 9) == self.num_mines:
                done = True
                reward = self.rewards['clear']

        self.done = done
        next_state = self.playerfield
        return next_state, reward, done

    def check_boundary(self, x, y):
        return 0 <= x < self.grid_size_X and 0 <= y < self.grid_size_Y

    def auto_reveal_tiles(self, x, y):
        queue = deque([(x, y)])

        while queue:
            cx, cy = queue.popleft()
            self.visited.add((cx, cy))
            self.playerfield[cx, cy] = self.minefield[cx, cy]

            # (cx, cy) 주변 8개 타일 확인
            if self.minefield[cx, cy] == 0: # 방문하지 않았으면 open
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        # 인덱스가 게임판 범위 내에 있는지 확인
                        if self.check_boundary(nx, ny) and (nx, ny) not in self.visited and (nx, ny) not in queue:  # nonvisited 주위 타일 큐에 추가
                            queue.append((nx, ny))

    def render(self):
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
