class Environment:
    def __init__(self):
        self.grid_size_X = 9
        self.grid_size_Y = 9
        self.num_mines = 10

        # 실제 정답 minefield 초기화
        self.minefield = np.zeros((self.grid_size_X, self.grid_size_Y), dtype=int)

        # 실제 정답 playerfield 초기화: hidden(-1) 상태로
        self.playerfield = np.full((self.grid_size_X, self.grid_size_Y), -1, dtype=int)

        self.state_size = self.minefield.size  # (9*9 = 81 반환)

        self.explode = False  # 폭발 여부
        self.done = False  # 게임 끝 여부
        self.first_move = True  # 처음 open하는지 여부
        self.visit_count = {}  # 각 타일의 방문 횟수를 기록

        self.rewards = {'explode' : -1, 'nonprogress' : -1,'progress' : 0.3, 'guess' : 0.1, 'clear' : 2}

    def reset(self):
        self.minefield = np.zeros((self.grid_size_X, self.grid_size_Y), dtype=int)
        self.playerfield = np.full((self.grid_size_X, self.grid_size_Y), -1, dtype=int)  # Hidden 상태로 초기화

        self.explode = False
        self.done = False
        self.first_move = True

        self.visit_count = {}  # 각 타일의 방문 횟수를 초기화

        self.place_mines()

        return list(self.playerfield)

    def place_mines(self):
        mines_placed = 0

        # num.mines만큼 임의의 좌표에 지뢰 심기
        while mines_placed < self.num_mines:
            x = random.randint(0, self.grid_size_X - 1)
            y = random.randint(0, self.grid_size_Y - 1)

            if self.minefield[x, y] == 0:  # 아직 지뢰가 놓이지 않은 좌표
                self.minefield[x, y] = -2
                mines_placed += 1

        # 지뢰 없는 좌표: 인접 지뢰 개수 세기
        for x in range(self.grid_size_X):
            for y in range(self.grid_size_Y):
                if self.minefield[x, y] == -2:  # 이미 지뢰가 놓인 좌표 제외
                    continue
                # 지뢰가 없는 좌표에 대해 주변 지뢰 개수(0~8)를 state로 설정
                self.minefield[x, y] = self.count_adjacent_mines(x, y)

    def count_adjacent_mines(self, x, y):
        count = 0
        # (x,y) 주변 지뢰 개수
        for i in range(max(0, x - 1), min(self.grid_size_X, x + 2)):  # 좌우 탐색 (경계 고려)
            for j in range(max(0, y - 1), min(self.grid_size_Y, y + 2)):  # 상하 탐색 (경계 고려)
                if self.minefield[i, j] == -2:
                    count += 1
        return count

    def count_adjacent_hidden(self, x, y):
        count = 0
        # (x,y) 주변 지뢰 개수
        for i in range(max(0, x - 1), min(self.grid_size_X, x + 2)):  # 좌우 탐색 (경계 고려)
            for j in range(max(0, y - 1), min(self.grid_size_Y, y + 2)):  # 상하 탐색 (경계 고려)
                if self.playerfield[i, j] == -2:
                    count += 1
        return count

    def step(self, action):
        x, y = divmod(action, self.grid_size_X)  # 1차원 인덱스를 2차원 좌표로 변환

        reward = 0
        done = False

        # explode: 지뢰 선택 시 done
        if self.minefield[x, y] == -2:
            self.playerfield[x, y] = self.minefield[x, y]  # 타일 열기
            self.explode = True
            done = True
            reward = self.rewards['explode']

        # 지뢰를 선택하지 않은 경우
        else:
          # nonprogress: 선택한 좌표 (x,y)가 이미 방문된 경우
            if (x, y) in self.visit_count:
                # self.visit_count[(x, y)] += 0.1  # 방문 횟수 증가
                # reward = self.rewards['nonprogress'] * self.visit_count[(x, y)]
                reward = self.rewards['nonprogress']
          # 선택한 좌표 (x, y)가 처음 방문된 경우
            else:
                self.visit_count[(x, y)] = 1  # 방문 횟수 초기화
                self.playerfield[x, y] = self.minefield[x, y]  # 타일 열기
                # guess: 주위 타일이 전부 hidden
                if self.count_adjacent_hidden(x, y) == 8:
                      reward = self.rewards['guess']
                # progress: 주위 opened 타일 존재
                else:
                      reward = self.rewards['progress']
                # open한 타일이 0이면 주위 타일 open
                if self.playerfield[x, y] == 0:
                  self.auto_reveal_tiles(x, y)  # (x, y) 주변 타일 열기

            # clear: 모든 hidden 타일이 지뢰만 남아 있는 경우 승리
            if np.count_nonzero(self.playerfield == -1) == self.num_mines:
                done = True
                reward = self.rewards['clear']

        self.done = done
        next_state = self.playerfield
        return next_state, reward, done

    def check_boundary(self, x, y):
        return 0 <= x < self.grid_size_X and 0 <= y < self.grid_size_Y

    def auto_reveal_tiles(self, x, y):  # BFS
        queue = deque([(x, y)])
        self.visited = set()

        while queue:
            cx, cy = queue.popleft()
            self.visited.add((cx, cy))  # (cx, cy) 방문 표시
            self.playerfield[cx, cy] = self.minefield[cx, cy]  # (cx, cy) 타일 열기
            self.visit_count[(cx, cy)] = self.visit_count.get((cx, cy), 0) + 1  # 방문 횟수 기록

            # (cx, cy) 주변 8개 타일 확인, 범위 내에 있으면 큐에 insert
            if self.minefield[cx, cy] == 0: # 방문하지 않았으면 open
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        # 인덱스가 게임판 범위 내에 있는지 확인
                        if self.check_boundary(nx, ny) and (nx, ny) not in self.visited and (nx, ny) not in queue:  # nonvisited 주위 타일 큐에 추가
                            queue.append((nx, ny))

    def render(self):  # 인수 설정
        for x in range(self.grid_size_X):
            for y in range(self.grid_size_Y):
                tile = self.playerfield[x, y]
                if tile == -1:
                    print('.', end=' ')
                elif tile == -2:
                    print('X', end=' ')
                else:
                    print(tile, end=' ')
                if y == self.grid_size_Y - 1:
                    print()
        print('\n')
