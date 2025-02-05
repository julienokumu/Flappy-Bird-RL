import torch # deep learning
import torch.nn as nn # neural network module
import pygame
import numpy as np # mathematical operations
import torch.optim as optim # optimization algorithms
import random
import os
from collections import deque # data structure for queue operations

# initialize pygame
pygame.init()

# game constants
screen_width, screen_height = 512, 512
bird_x = 50 # fixed horizontal position of bird
pipe_width = 52
gravity = 1 # effect of birds movements downwards
jump_strength = -10 # upward velocity of bird
pipe_speed = 3 # speed of movement of pipes to the left
pipe_gap = 260
white = (255, 255, 255)

# load images
bird_imgs = [pygame.image.load(f"bird{i}.png") for i in range(3)] # load three diff bird images
pipe_img = pygame.image.load("pipe.png")
background_img = pygame.image.load("background.png")

# flappy bird environment
class FlappyBirdEnv:
    def __init__(self):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Flappy Bird RL | Julien")
        self.clock = pygame.time.Clock()
        self.reset()
        self.bird_index = 0 # initial bird cycle img
        self.frame_count = 0 # counter to track frames

    def reset(self):
        self.bird_y = screen_height // 2 # bird at middle of screen
        self.bird_velocity = 0
        self.pipe_x = screen_width # starts pipe at the right side of screen

        # randomly set top pipe height while maintaining gap
        self.top_pipe_height = random.randint(50, screen_height - pipe_gap - 50)
        self.bottom_pipe_height = self.top_pipe_height + pipe_gap
        self.score = 0 # initial score
        return self.get_state() # return the initial game state
    
    def step(self, action):
        """
        updates game state based on action taken
        args:
            action(int): 0 for no action(fall), 1 for jumping
        returns:
            state(numpy array): current game state
            reward(int): reward for action taken
            done(bool): whether the game is over
        """
        if action == 1:
            self.bird_velocity = jump_strength # apply jump velocity

        self.bird_velocity += gravity # apply gravity
        self.bird_y += self.bird_velocity # update birds position
        self.pipe_x -= pipe_speed # move pipe to the left

        # if pipe moves off screen, reset its position and generate new pipes
        if self.pipe_x < -pipe_width:
            self.pipe_x = screen_width # reset pipe to the right
            # generate new pipes
            self.top_pipe_height = random.randint(50, screen_height - pipe_gap - 50)
            self.bottom_pipe_height = self.top_pipe_height + pipe_gap
            self.score += 1 # increase score if bird passes pipe

        # check if game is over
        done = self.bird_y > screen_height or self.bird_y < 0 or (
            self.pipe_x < bird_x + 34 and not (self.top_pipe_height < self.bird_y < self.bottom_pipe_height)
        )
        # reward system +100 for staying alive, large negative reward for losing
        reward = 100 if not done else -10000

        self.render() # render game after updating state
        return self.get_state(), reward, done # return new state, reward and game status
    
    def get_state(self):
        """
        returns current game state as a numerical representation
        state:
            birds vertical position
            bird horizontal velocity
            pipes horizontal position
            top pipes height
        """
        return np.array([self.bird_y, self.bird_velocity, self.pipe_x, self.top_pipe_height], dtype=np.float32)
    
    def render(self):
        # renders all game visuals and images
        self.screen.blit(background_img, (0, 0))

        # top pipe flipped vertically
        top_pipe = pygame.transform.flip(pipe_img, False, True)
        self.screen.blit(top_pipe, (self.pipe_x, self.top_pipe_height - top_pipe.get_height()))

        # bottom pipe
        self.screen.blit(pipe_img, (self.pipe_x, self.bottom_pipe_height))

        # bird animation
        self.frame_count +=1 
        if self.frame_count % 5 == 0: # change bird image after every 5 seconds
            self.bird_index = (self.bird_index + 1) % len(bird_imgs) # cycle through bird images
        self.screen.blit(bird_imgs[self.bird_index], (bird_x, self.bird_y)) # draw bird

        pygame.display.update() # update game display
        self.clock.tick(30) # limit game to 30 frames per second

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # input = state_size, output = 24 neurons
        self.fc1 = nn.Linear(state_size, 24)
        # input 24 = neurons, output = 24 neurons
        self.fc2 = nn.Linear(24, 24)
        # input 24 neurons, output = action_size
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        # pass input through first layer with ReLU
        x = torch.relu(self.fc1(x))
        # pass through second layer with ReLU
        x = torch.relu(self.fc2(x))
        # pass through last layer with no activation function as we want raw q values for each action
        return self.fc3(x)
    
# DQN agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # number of features in environment
        self.action_size = action_size # number of possible actions
        self.memory = deque(maxlen=2000) # max experiences in a replay buffer
        self.gamma = 0.95 # dicount factor for future rewards
        self.epsilon = 1.0 # initial exploration rate (100%) for randomness
        self.epsilon_min = 0.01 # min exploration rate
        self.epsilon_decay = 0.995 # rate of decay towards min epsilon
        self.learning_rate = 0.001 # learning rate for optimizer
        self.model = DQN(state_size, action_size) # initialize model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # mean squared error loss for training

    def remember(self, state, action, reward, next_state, done):
        # store experiences
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # decide action using epsilon
        if np.random.rand() <= self.epsilon:
            # choose a random action with probability epsilon
            return random.randrange(self.action_size)
        # convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): # disable gradient  calculation for inference
            q_values = self.model(state) # get q values from model
        return np.argmax(q_values.numpy()) # select action with highest q value
    
    def replay(self, batch_size):
        # train model using a batch fo past experiences
        if len(self.memory) < batch_size:
            return # do nothing if there are not enough experiences
        minibatch = random.sample(self.memory, batch_size) # sample a batch from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward # set targte to immediate reward
            if not done: # if episode is not finished, add discounted future reward
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)) # get current q values
            target_f[0][action] = target # update q value for action taken
            self.optimizer.zero_grad() # clear previous gradients
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state).unsqueeze(0))) # compute loss
            loss.backward() #backpropagate loss
            self.optimizer.step() # update model parameters
        # reduce episilon to decrease randomness over time and make use of exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# training loop
def train():
    env = FlappyBirdEnv() # intiialize env
    agent = Agent(state_size=4, action_size=2)
    episodes = 1000 # number of training episodes
    batch_size = 32 # number of experinces per training step

    for episode in range(episodes): # loop through each episode
        state = env.reset() # reset env and get intial state
        total_reward = 0 # track total reward for episode
        done = False # track if episode is finished
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state) # get action from agent
            next_state, reward, done = env.step(action) # take action and observe new state and reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state # update current state
            total_reward += reward # accumumlate reward
        agent.replay(batch_size) # train agent using replay memory
        print(f"Episode {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}") # print episode summary

if __name__ == "__main__":
    train() # start training
    print("Training Complete.")


    
