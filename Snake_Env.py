# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Project: Reinforcement Learning on Snake
# Author: Cai Ruikai
# Date: 2020.10.10
# ------------------------------------------------------------------------------------------------------------------------------------------------------
import sys
from PIL import Image
import random
import pygame
import numpy as np
from collections import deque
from pygame.locals import *
import math

default_config = {
    'SCREEN_WIDTH': 200,  # 屏幕宽度
    'SCREEN_HEIGHT': 200,  # 屏幕高度
    'BLOCK_SIZE': 20,  # 方格大小

    'CROSS_BOUNDARY': False,  # 是否允许穿过边界

    'BACKGROUND_COLOR': (60, 60, 60),  # 背景颜色
    'SNAKE_COLOR': (0, 160, 100),  # 蛇的颜色
    'FOOD_COLOR': (240, 240, 240),  # 水果的颜色
    'OBSTACLE_COLOR': (255, 0, 0),  # 障碍物颜色

    'OBSTACLE': False,
    'OBSTACLE_NUM': 2,  # 障碍物数量
    'OBSTACLE_FRESH': False,  # 是否刷新障碍物位置
    'OBSTACLE_FRESH_RATE': 30  # 障碍物刷新频率
}


class Snake_Env:
    def __init__(self, config=None):
        # use default config
        if not config:
            config = default_config
        self.clock = pygame.time.Clock()

        # game environment setting
        self._SCREEN_WIDTH = config['SCREEN_WIDTH']
        self._SCREEN_HEIGHT = config['SCREEN_HEIGHT']
        self._BLOCK_SIZE = config['BLOCK_SIZE']
        self._X_AREA = (0, self._SCREEN_WIDTH // self._BLOCK_SIZE - 1)
        self._Y_AREA = (0, self._SCREEN_HEIGHT // self._BLOCK_SIZE - 1)

        self._CROSS_BOUNDARY = config['CROSS_BOUNDARY']

        self._OBSTACLE_NUM = config['OBSTACLE_NUM']
        self._OBSTACLE_FRESH = config['OBSTACLE_FRESH']
        self._OBSTACLE_FRESH_RATE = config['OBSTACLE_FRESH_RATE']

        self._BACKGROUND_COLOR = config['BACKGROUND_COLOR']
        self._SNAKE_COLOR = config['SNAKE_COLOR']
        self._OBSTACLE_COLOR = config['OBSTACLE_COLOR']
        self._FOOD_COLOR = config['FOOD_COLOR']
        self._OBSTACLE=config['OBSTACLE']

        # game status init
        self._step = 0
        self._score = 0
        self._step_reward = 0
        self._game_over = 0

        # pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self._SCREEN_WIDTH, self._SCREEN_HEIGHT))
        pygame.display.set_caption('Snake     Score:0')

        # snake,fruits,obstacles init
        self._snake, self._death_pos, self._foods, self._obstacles = deque(), (-1, 0), list(), list()
        self._init_spfo()
        self.render()
        pygame.display.update()

        # obs_dim, act_dim
        self._obs_dim = (self._SCREEN_WIDTH, self._SCREEN_HEIGHT, 3)
        self._act_dim = 4

    def _init_spfo(self):
        # snake
        for i in range(3):
            self._snake.append((3 - i, 0))
        # pos
        self._death_pos = (-1, 0)
        # foods
        self._foods.append(self._generate_xy())
        # obstacles
        if self._OBSTACLE:
            for i in range(self._OBSTACLE_NUM):
                self._obstacles.append(self._generate_xy())

    def _generate_xy(self):
        x = random.randint(self._X_AREA[0], self._X_AREA[1])
        y = random.randint(self._Y_AREA[0], self._Y_AREA[1])
        while (x, y) in set(list(self._snake) + self._obstacles + self._foods):
            x = random.randint(self._X_AREA[0], self._X_AREA[1])
            y = random.randint(self._Y_AREA[0], self._Y_AREA[1])
        return x, y

    def _refresh_food(self, eaten_food):
        if eaten_food == 0:
            self._foods[0] = self._generate_xy()
        # if eaten_food == 1:
        #     self._foods.pop()
        # if self._step % self._X_AREA[1]*2 == 0:
        #     self._foods = self._foods[:1]
        #     self._foods.append(self._generate_xy())
        # if self._step % self._X_AREA[1] > self._X_AREA[1] *0.9:
        #     self._foods = self._foods[:1]

    def _refresh_obstacle(self):
        if not self._OBSTACLE: return
        if self._step % self._OBSTACLE_FRESH_RATE == 0:
            self._obstacles = list()
            for i in range(self._OBSTACLE_NUM):
                self._obstacles.append(self._generate_xy())

    def _print_over(self):
        font = pygame.font.Font(None, 50)
        fwidth, fheight = font.size('GAME OVER')
        imgtext = font.render('GAME OVER', True, (255, 0, 0))
        self.screen.blit(imgtext, (self._SCREEN_WIDTH // 2 - fwidth // 2, self._SCREEN_HEIGHT // 2 - fheight // 2))

    def _move_snake(self, action):
        env_action = ((-1, 0), (1, 0), (0, -1), (0, 1))
        reward, eaten_food = -0.15, -1
        # judge whether action is legal
        contrary = {(0, 1): (0, -1), (0, -1): (0, 1), (1, 0): (-1, 0), (-1, 0): (1, 0)}
        action = env_action[action]
        if action == self._death_pos:
            self._game_over = 1
            # print('\nDeath: action is legal')
        else:
            next_head = (self._snake[0][0] + action[0], self._snake[0][1] + action[1])
            # obstacles
            if next_head in self._obstacles:
                self._game_over = 1
                # print('\nDeath: obstacles')
            # boundary
            if self._CROSS_BOUNDARY:
                if next_head[0] < 0:
                    next_head = (self._X_AREA[1], next_head[1])
                elif next_head[0] > self._X_AREA[1]:
                    next_head = (0, next_head[1])
                elif next_head[1] < 0:
                    next_head = (next_head[0], self._Y_AREA[1])
                elif next_head[1] > self._Y_AREA[1]:
                    next_head = (next_head[0], 0)
            else:
                if next_head[0] < 0 or next_head[0] > self._X_AREA[1] or next_head[1] < 0 or next_head[1] > \
                        self._Y_AREA[
                            1]:
                    self._game_over = 1
                    # print('\nDeath: boundary')
            # body:
            if next_head in self._snake:
                self._game_over = 1
                # print('\nDeath: body')

            # fruit
            if next_head in self._foods:
                eaten_food = self._foods.index(next_head)
                # reward += 10 if eaten_food == 0 else 50
                reward += 1

            else:
                self._snake.pop()

            self._snake.appendleft(next_head)
        if not self._game_over:
            self._death_pos = contrary[action]
            dis = (math.sqrt(pow((self._foods[0][0] - next_head[0]), 2) + pow((self._foods[0][1] - next_head[1]), 2)))
            dis_reward = (1 / max(1.0, dis)) * 0.5
            reward += dis_reward
        else:
            reward = -1
        self._score += eaten_food+1 if eaten_food<1 else 5
        return reward, eaten_food

    def render(self):
        # draw background
        self.screen.fill(self._BACKGROUND_COLOR)
        # draw gird and x-axis
        for x in range(self._BLOCK_SIZE, self._SCREEN_WIDTH, self._BLOCK_SIZE):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self._SCREEN_HEIGHT), 1)
        # draw gird and y-axis
        for y in range(self._BLOCK_SIZE, self._SCREEN_HEIGHT, self._BLOCK_SIZE):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self._SCREEN_WIDTH, y), 1)
        # draw food
        for index, food in enumerate(self._foods):
            food_color = self._FOOD_COLOR if index == 0 else (255, 215, 0)
            pygame.draw.circle(self.screen, food_color,
                               (food[0] * self._BLOCK_SIZE + self._BLOCK_SIZE // 2,
                                food[1] * self._BLOCK_SIZE + self._BLOCK_SIZE // 2),
                               self._BLOCK_SIZE // 2, 0)
        # draw obstacle
        for obs in self._obstacles:
            pygame.draw.rect(self.screen, self._OBSTACLE_COLOR,
                             ((obs[0] * self._BLOCK_SIZE, obs[1] * self._BLOCK_SIZE),
                              (self._BLOCK_SIZE, self._BLOCK_SIZE)),
                             0)
        # draw snake
        for index, node in enumerate(self._snake):
            if index == 0:
                pygame.draw.circle(self.screen, (220,20,20),
                                   (node[0] * self._BLOCK_SIZE + self._BLOCK_SIZE // 2,
                                    node[1] * self._BLOCK_SIZE + self._BLOCK_SIZE // 2),
                                   self._BLOCK_SIZE // 2, 0)
            else:
                pygame.draw.rect(self.screen, self._SNAKE_COLOR,
                                 ((node[0] * self._BLOCK_SIZE, node[1] * self._BLOCK_SIZE),
                                  (self._BLOCK_SIZE, self._BLOCK_SIZE)),
                                 0)
        if self._game_over:
            self._print_over()

        pygame.display.set_caption('Score:{:.3f}'.format(self._score))
        pygame.display.update()
        # clock = pygame.time.Clock()
        # clock.tick(10)

    def init(self):
        self.__init__()
        obs = pygame.surfarray.array3d(pygame.display.get_surface()).transpose((1, 0, 2))
        return obs

    def obs_dim(self):
        return self._obs_dim

    def act_dim(self):
        return self._act_dim

    def reset(self):
        return self.init()

    def frame_step(self, action):
        self._step += 1

        self._step_reward, eaten_food = self._move_snake(action)
        self._refresh_food(eaten_food)
        if self._OBSTACLE_FRESH:
            self._refresh_obstacle()

        self.render()
        obs = pygame.surfarray.array3d(pygame.display.get_surface()).transpose((1, 0, 2))

        return obs, self._step_reward, self._score, self._game_over, self.get_game_info()

    @staticmethod
    def get_human_action():
        action = None
        while not action:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        return 0
                    if event.key == pygame.K_RIGHT:
                        return 1
                    if event.key == pygame.K_UP:
                        return 2
                    if event.key == pygame.K_DOWN:
                        return 3
                else:
                    pass

    def get_game_info(self):
        info = {'step': self._step,
                'snake': list(self._snake),
                'obstacles': self._obstacles,
                'foods': self._foods}
        return info

    def print_game_info(self):
        info = self.get_game_info()
        print('\ncurrent step:{} reward :{} game score : {}'.format(info['step'], self._step_reward, self._score))
        for k, v in info.items():
            if k == 'count':
                continue
            print(k, v)


def demo():
    env = Snake_Env()
    env.init()
    env.render()
    game_over = False
    while not game_over:
        human_action = env.get_human_action()
        obs, step_reward, game_score, game_over, info = env.frame_step(human_action)
        print(obs.shape)
        env.print_game_info()
        if game_over:
            print(game_score)
            game_over = False
            env.reset()
            env.render()


if __name__ == '__main__':
    demo()
