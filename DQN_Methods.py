# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Project: Reinforcement Learning on Snake
# Author: Cai Ruikai
# Date: 2020.10.29
# ------------------------------------------------------------------------------------------------------------------------------------------------------

from abc import ABC
import torch
import torch.nn as nn
import random
import tqdm
from utils import *
from Snake_Env import *
from PIL import Image
from collections import deque, OrderedDict

# Double DQN method
DQN_Default_Conf = {
    'memory_size': 100000,
    'learn_start': 5000,
    'batch_size': 128,
    'learn_freq': 2,
    'target_update_freq': 4,
    'clip_norm': 5,
    'learning_rate': 0.001,
    'eps': 0.2,
    'max_train_iteration': 5000000,
    'reward_threshold': 5000,
    'max_episode_length': 500,
    'gamma': 0.1,
    'evaluate_int': 1,
}


class DQN_Network(nn.Module, ABC):
    def __init__(self, obs_dim=(520, 520, 3), act_dim=4):
        super(DQN_Network, self).__init__()

        self.input_shape = obs_dim
        self.num_actions = act_dim
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, 3, 2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(16, 32, 3)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(32, 32, 3, 2)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(32, 16, 3)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(16, 1, 3, 2)),
            ('relu5', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(484, 128)),
            ('relu7', nn.ReLU()),
            ('linear2', nn.Linear(128, act_dim))
        ]))
        # print('Network Build Done:\n', self.model)

    def forward(self, observation):
        observation = process_state(observation)
        return self.model(observation)


class DQN_Agent(DQN_Network, ABC):
    def __init__(self):
        super(DQN_Agent, self).__init__()

    def load_weights(self, weights=None):
        if weights:
            self.mode.load_state_dict(weights)
        return self.model


class DQN_Method:
    def __init__(self, config=None):
        if not config:
            config = DQN_Default_Conf
        # parameters
        self.best_scores = 0
        self.learn_freq = config["learn_freq"]
        self.learn_start = config["learn_start"]
        self.learning_rate = config["learning_rate"]
        self.target_update_freq = config["target_update_freq"]
        self.memory = ReplayMemory(capacity=config["memory_size"], learn_start=config["learn_start"])

        self.batch_size = config["batch_size"]
        self.max_train_iteration = config["max_train_iteration"]
        self.max_episode_length = config["max_episode_length"]

        self.eps = config["eps"]
        self.gamma = config["gamma"]
        self.clip_norm = config["clip_norm"]
        self.evaluate_int = config["evaluate_int"]
        self.reward_threshold = config["reward_threshold"]

        self.total_step = 0
        self.step_since_update = 0
        self.step_since_evaluate = 0

        # create environment
        self.env = Snake_Env()
        self.obs_dim = self.env.obs_dim()
        self.act_dim = self.env.act_dim()

        # create double DQN network
        self.network = DQN_Network(self.obs_dim, self.act_dim)
        self.network.eval()

        self.target_network = DQN_Network(self.obs_dim, self.act_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.network.eval()

        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def compute_action(self, observation, eps=None):
        values = self.target_network(observation)
        values = values.detach().numpy()
        if not eps:
            eps = self.eps
        action = np.argmax(values) if np.random.random() > eps else np.random.choice(self.act_dim)

        return action

    def prepare_memory(self):
        pbar = tqdm.tqdm(total=self.learn_start, desc="preparing replay memory")
        while len(self.memory) < self.learn_start:
            env = Snake_Env()
            current_state = env.reset()
            act = self.compute_action(current_state)
            for t in range(self.max_episode_length):

                next_state, reward, score, game_over, other_info = env.frame_step(act)
                transition = (current_state, act, reward, next_state, game_over)
                # self.test(transition)

                self.memory.push(transition)
                pbar.update()
                current_state = next_state
                act = self.compute_action(current_state)
                if game_over:
                    break
        pbar.close()

    def test(self, transition):
        current_state, act, reward, next_state, game_over = transition
        current_state = Image.fromarray(np.uint8(current_state))
        current_state.show(title="current_state")
        next_state = Image.fromarray(np.uint8(next_state))
        next_state.show(title="next_state")
        print(act, reward, game_over)
        input()

    def train(self,  pre_trained=None):
        if pre_trained:
            self.network=torch.load(pre_trained)
            self.target_network=torch.load(pre_trained)
            self.best_scores=float(pre_trained.split('_')[1])
            self.total_step=int(pre_trained.split('_')[2][:-3])

        self.prepare_memory()
        print('Start Training')
        # train network in max_train_iteration
        for train_iteration in range(self.max_train_iteration):
            current_state = self.env.reset()
            act = self.compute_action(current_state)
            stat = {"loss": []}

            # each train iteration has max episode length
            for t in range(self.max_episode_length):

                next_state, reward, score, game_over, other_info = self.env.frame_step(act)
                transition = (current_state, act, reward, next_state, game_over)
                # self.test(transition)

                self.memory.push(transition)

                self.total_step += 1
                self.step_since_update += 1

                if game_over:
                    break

                current_state = next_state
                act = self.compute_action(current_state)

                if t % self.learn_freq != 0: continue

                states, actions, rewards, next_states, not_done_mask = self.memory.get_batch(self.batch_size)
                # image1=Image.fromarray(states[0])
                # image1.show('1')
                # image2 = Image.fromarray(next_states[0])
                # image2.show('2')
                # print(actions[0],rewards[0],not_done_mask[0])
                # input()

                with torch.no_grad():
                    Q_t_plus_one_max = self.target_network(next_states).max(1)[0]
                    Q_t_plus_one = Q_t_plus_one_max * not_done_mask
                    Q_target = rewards + self.gamma * Q_t_plus_one

                self.network.train()
                Q_t = self.network(states)
                Q_t = Q_t.gather(1, actions).squeeze()

                assert Q_t.shape == Q_target.shape, print(Q_t.shape, Q_target.shape)

                # Update the network
                self.optimizer.zero_grad()
                loss = self.loss(input=Q_t, target=Q_target)
                loss_value = loss.item()
                stat['loss'].append(loss_value)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm)
                self.optimizer.step()
                self.network.eval()

            # update target network
            if self.step_since_update > self.target_update_freq:
                print('\nCurrent train iteration:{} Current memory:{}'.format(train_iteration,len(self.memory)))
                print('Current step:{},{} steps has passed since last update,Now update behavior policy'.format(self.total_step, self.step_since_update))

                self.step_since_update = 0
                self.target_network.load_state_dict(self.network.state_dict())
                self.target_network.eval()

                # evaluate and save network
                self.step_since_evaluate += 1
                if self.step_since_evaluate >= self.evaluate_int:
                    self.step_since_evaluate = 0
                    eva_score, eva_length, eva_reward = self.evaluate()
                    print("best score:{},loss:{:.2f},episode length:{:.2f},evaluate score:{:.2f},evaluate reward:{:.2f}"
                          .format(self.best_scores,np.mean(stat["loss"]),eva_length ,eva_score,eva_reward))

                    torch.save(self.target_network, 'last.pt')
                    # save best network
                    if eva_score > self.best_scores and eva_score > 1:
                        print('save model of performance:', eva_score)
                        self.best_scores = eva_score
                        torch.save(self.target_network, 'best_{:.2f}_{}.pt'.format(eva_score, self.total_step))

    def evaluate(self, weights=None, num_episodes=30, episodes_len=150):
        pbar = tqdm.tqdm(total=num_episodes, desc="evaluating")
        env = Snake_Env()
        policy = self.target_network
        if weights:
            policy.load_state_dict(weights)
        rewards = []
        epo_len = []
        scores = []
        for i in range(num_episodes):
            obs = env.reset()
            with torch.no_grad():
                act = np.argmax(policy(obs).detach().numpy())
            epo = 0
            score = 0
            ep_reward=0
            for t in range(episodes_len):
                next_state, reward, score, game_over, other_info = env.frame_step(act)
                act = np.argmax(policy(next_state).detach().numpy())
                if game_over:
                    break
                epo += 1
                ep_reward+=reward
            epo_len.append(epo)
            rewards.append(ep_reward)
            scores.append(score)
            pbar.update()
        pbar.close()
        return np.mean(scores), np.mean(epo_len),np.mean(rewards)


def demo():
    pass


if __name__ == '__main__':
    DQN = DQN_Method()
    DQN.train()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DQN.network.to(device)
    # from torchsummary import summary
    # summary(model,(3,200,200))