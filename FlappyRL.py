import torch # for deep learning
import torch.nn as nn # for neural network building
import torch.optim as optim # optimization algorithm
import pygame # game visualization and graphics
import os # interacting with operating system
from collections import deque # for efficient data structure for queue operations
import random # for random number generation
import numpy as np # mathematical matrix operations

# intialize pygame
pygame.init()

# game constants
screen_width, screen_height = 512, 512 # game window
bird_x = 50 # fixed horizontal position of bird
pipe_width = 52
gravity = 1 # effect of birds downward movement
jump_strength = -10 # upward velocity when bird jumps
pipe_speed = 3 # speed of pipes moving leftward
pipe_gap = 200 # gap between top and bottom pipes
white = (255, 255, 255)

# load game assets
bird_images = [pygame.image.load(f"bird{i}.png") for i in range(3)] # load different frames
pipe_image = pygame.image.load("pipe.png")
background_image = pygame.image.load("background.png")

# define flappybird gym-like environment
class FlappyBirdEnv:
    def __init__(self):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Flappy Bird RL | Julien")
        self.clock = pygame.time.Clock() # clock object
        self.reset()
        self.bird_index = 0 # index to cycle throught bird images for animation
        self.frame_count = 0 # counter to track frames for animation

    def reset(self): # returns intial state of game
        self.bird_y = screen_height // 2 # starts bird in the middle of screen
        self.bird_velocity = 0 # start with 0 vertical movement
        self.pipe_x = screen_width # start pipe at the right edge of the screen

        # randomly set the top pipe height while maintaining the gap between the pipes
        self.pipe_top_height = random.randint(50, screen_height - pipe_gap - 50)
        self.pipe_bottom_height = self.pipe_top_height + pipe_gap # bottom pipe position
        self.score = 0 # start with score 0
        return self.get_state() # return intial game state

    def step(self, action):
        """
        updates the game state based on action taken
        args: action(int): 0 for no action(fall), 1 for jumping

        returns: state(numpy array): current game state, reward(int): reward for action, done(bool): whether the game is over
        """
        if action == 1:
            self.bird_velocity = jump_strength # apply jump velocity

        self.bird_velocity += gravity # increase downward speed
        self.bird_y += self.bird_velocity # update bird position
        self.pipe_x -= pipe_speed # move pipe leftward

        # if pipe moves off screen reset its position and generate new pipe heights
        if self.pipe_x < -pipe_width:
            self.pipe_x = screen_width # reset pipe to the right side
            # generate new random height for the top pipe maintaining a valid gap
            self.pipe_top_height = random.randint(50, screen_height - pipe_gap - 50)
            self.pipe_bottom_height = self.pipe_top_height + pipe_gap
            self.score += 1 # increase score for successfully passing a pipe

        # check if the game is over
        done = self.bird_y > screen_height or self.bird_y < 0 or (
            self.pipe_x < bird_x + 34 and not (self.pipe_top_height < self.bird_y < self.pipe_bottom_height)
        )
        # reward system: +1 for staying alive, large negative reward for losing
        reward = 1 if not done else -100

        self.render() # render game after updating the state
        return self.get_state(), reward, done # return the new game state, reward and game state
    
    def get_state(self):
        # returns current game state as a numerical representation
        # state consists of birds vertical pos, birds vertical velocity, pipes horizontal position, top pipes height
        return np.array([self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_top_height], dtype=np.float32)

    def render(self): # renders game visuals
        self.screen.blit(background_image, (0, 0))
        
        # draw top pipe(flipped vertically)
        top_pipe = pygame.transform.flip(pipe_image, False, True)
        self.screen.blit(top_pipe, (self.pipe_x, self.pipe_top_height - top_pipe.get_height()))

        # draw the bottom pipe
        self.screen.blit(pipe_image, (self.pipe_x, self.pipe_bottom_height))

        # handle bird animation(cycle through images every few frames)
        self.frame_count += 1
        if self.frame_count % 5 == 0: # change bird image every 5 frames
            self.bird_index = (self.bird_index + 1) % len(bird_images) # cycle through bird images
        self.screen.blit(bird_images[self.bird_index], (bird_x, self.bird_y)) # draw the bird

        pygame.display.update() # update the game display
        self.clock.tick(30) # limit the game to 30 frames per second

# Deep Q Network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24) # first layer: input = state_size output = 24 neurons
        self.fc2 = nn.Linear(24, 24) # second layer: input = 24 neurons, output = 24 neurons
        self.fc3 = nn.Linear(24, action_size) # thrid layer: input = 24 neurons, output = action_size

    def forward(self, x):
        # pass input throught first layer and apply ReLU activation function
        x = torch.relu(self.fc1(x))
        # pass through second layer with ReLU
        x = torch.relu(self.fc2(x))
        # pass through final layer(no activation function, raw q values for each possible action)
        return self.fc3(x)

# DQN agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # number of featers in the state presentation
        self.action_size = action_size # number of possible actions
        self.memory = deque(maxlen=2000) # replay memory buffer with a max size of 2000 experiences
        self.gamma = 0.95 # discount factor for future rewards
        self.epsilon = 1.0 # initial exploration rate(100% random actions at start)
        self.epsilon_min = 0.01 # minimum exploration rate, ensure randomness
        self.epsilon_decay = 0.995 # rate of exploration decays, reducing randomness
        self.learning_rate = 0.001 # learning rate for optimizer
        self.model = DQN(state_size, action_size) # initialize DQN model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # adam optimizer
        self.criterion = nn.MSELoss() # Mean squared error loss function for training

    def remember(self, state, action, reward, next_state, done):
        # store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # decide action using epsilon greedy
        if np.random.rand() <= self.epsilon:
            # choose a random action with probability epsilon
            return random.randrange(self.action_size)
        # convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): # disable gradient calculation for inference
            q_values = self.model(state) # get q values from the model
        return np.argmax(q_values.numpy()) # select action with highest q value

    def replay(self, batch_size):
        # train the model using a batch of past experiences
        if len(self.memory) < batch_size:
            return # do nothing if there arent enough experiences yet
        minibatch = random.sample(self.memory, batch_size) # sample a batch from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward # set target to immediate reward
            if not done: # if episode is not finished add discounted future reward
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)) # get current q values
            target_f[0][action] = target # update q value for taken action
            self.optimizer.zero_grad() # clear previous gradients
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state).unsqueeze(0))) # compute loss
            loss.backward() # backpropagate loss
            self.optimizer.step() # update model parameters
        # reduce epsilon to decrease randomness over time for more exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# training loop
def train():
    env = FlappyBirdEnv() # initialize the flappy bird environment
    agent = Agent(state_size=4, action_size=2) # create agent
    episodes = 1000 # number of training episodes
    batch_size = 32 # number of experienes per training step

    for episode in range(episodes):
        state = env.reset() # reset environment and get initial state
        total_reward = 0 # track total reward for this episode
        done = False # Track if the episode has ended
        while not done:
            for event in pygame.event.get(): # handle pygame events
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state) # get action from agent
            next_state, reward, done = env.step(action) # take action and observe new state and reward
            agent.remember(state, action, reward, next_state, done) # store experiences in memory
            state = next_state # update current state
            total_reward += reward # accumulate reward
        agent.replay(batch_size) # train the agent using replay memory
        print(f"Episode {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}") # print episode summary

if __name__ == "__main__":
    train() # start training
    print("Training Complete")
        