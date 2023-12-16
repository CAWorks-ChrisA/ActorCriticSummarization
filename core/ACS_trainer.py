"""
ACS Trainer.

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(50000)):#1e6 or 1 million
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size): # single random sample, call multiple times for overlapping samples for CREST
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def getItem(self,index): # index-based call for the items
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )
    
    def dumpBuffer(self): # dump everything from the buffer.
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device)
        )


class ACSActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ACSActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.tanh(self.l3(a))
        return a
    
class ACSActorLastLayer(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ACSActorLastLayer, self).__init__()
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    def forward(self, logits):
        return self.max_action * torch.tanh(self.l3(logits))


class ACSCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACSCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        # q1 = self.l3(q1)

        return q1
    
class ACSCriticLastLayer(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACSCriticLastLayer, self).__init__()
        self.l3 = nn.Linear(256, 1)

    def forward(self, logits):
        q1 = self.l3(logits)
        return q1




class ACSTrainer:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=5e-5  # Small LR in TD3 is important to train in MetaDrive!
    ):
        self.actor = ACSActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.actorLL = ACSActorLastLayer(state_dim, action_dim, max_action).to(device)
        self.actorLL_target = copy.deepcopy(self.actorLL)
        self.actorLL_optimizer = torch.optim.Adam(self.actorLL.parameters(), lr=lr)

        self.critic = ACSCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.criticLL = ACSCriticLastLayer(state_dim, action_dim).to(device)
        self.criticLL_target = copy.deepcopy(self.criticLL)
        self.criticLL_optimizer = torch.optim.Adam(self.criticLL.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.coresetTrainData = []
        self.coresetWeights = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actorLL(self.actor(state)).cpu().data.numpy().flatten()

    def select_action_in_batch(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actorLL(self.actor(state)).cpu().data.numpy()
    
    def critic_generate_gradients(self,replay_buffer):
        # print (replay_buffer.size)
        print()
        print("Generating Gradients")
        gradients = []
        for i in range(replay_buffer.size):
            state, action, next_state, reward, not_done = replay_buffer.getItem(i)
            with torch.no_grad():
                # target_Q = self.critic_target(next_state.unsqueeze(0), self.actor_target(next_state.unsqueeze(0)))
                target_Q = self.criticLL_target(self.critic_target(next_state.unsqueeze(0), self.actorLL_target(self.actor_target(next_state.unsqueeze(0)))))
                target_Q = reward + (not_done * self.discount * target_Q)

            # Get current Q estimates
            
            # current_Q1 = self.critic(state.unsqueeze(0), action.unsqueeze(0))
            with torch.no_grad():
                logits = self.critic(state.unsqueeze(0), action.unsqueeze(0))
            current_Q1 = self.criticLL(logits)

            # TODO: Compute critic loss.
            # Hint: Compute the MSE for both critics and add them up.
            # critic_loss = None
            # pass
            criterion = nn.MSELoss()
            critic_loss = criterion(current_Q1,target_Q)

            # Optimize the critic
            # self.critic_optimizer.zero_grad()
            self.criticLL_optimizer.zero_grad()
            critic_loss.backward()
            # print(self.critic.l3.weight.grad.shape)
            # gradients.append(self.critic.l3.weight.grad)
            criticGrad = self.criticLL.l3.weight.grad

            logits = self.critic(state.unsqueeze(0),self.actorLL(self.actor(state)).unsqueeze(0))
            actor_loss = -self.criticLL(logits)
            self.actorLL_optimizer.zero_grad()
            actor_loss.backward()
            actorGrad = self.actorLL.l3.weight.grad.flatten().unsqueeze(0)
            # gradients.append(criticGrad)
            gradients.append(torch.cat([criticGrad,actorGrad],1))
        gradients = torch.vstack(gradients)
        print("Gradients generated with a shape of:",gradients.shape)
        self.critic_optimizer.zero_grad()
        self.criticLL_optimizer.zero_grad()
        return gradients
    
    def generate_random_subsets(self,replay_buffer,set_size,set_count):
        print()
        print("Generating random subset indexes")
        out = []
        for i in range(set_count):
            indexes = np.random.randint(0, replay_buffer.size, size=set_size)
            out.append(indexes)
        out = np.vstack(out)
        print("Generated subsets of shape: ", out.shape)
        return out

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        if self.total_it % 2500 == 1:
            random_subset_indexes = self.generate_random_subsets(replay_buffer,2000,15) # 2000,64,15, unlimited gets to 290 no drops
            gradients = self.critic_generate_gradients(replay_buffer)
            import core.submodular as submodular
            coresets = []
            self.coreset_weights = []
            for indexes in random_subset_indexes:
                (out_index,subset_weights,_,_,_,_)=submodular.get_orders_and_weights(64,gradients[indexes].cpu().detach().numpy(),"euclidean")
                coresets.append(indexes[out_index])
                self.coreset_weights.append(subset_weights)
            coresets = np.vstack(coresets)
            self.coreset_weights = np.vstack(self.coreset_weights)
            # convert coresets to data.
            self.coresetTrainData = []
            for indexes in coresets:
                trainDataTuple = replay_buffer.getItem(indexes)
                self.coresetTrainData.append(trainDataTuple)

        minibatchIndex = random.randint(0,len(self.coresetTrainData)-1)
        state,action,next_state,reward,not_done = self.coresetTrainData[minibatchIndex]
        minibatchWeight = self.coreset_weights[minibatchIndex] # -1 to stabilize training against singular item gradients (may be bad)


        # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        #----DO NOT USE FOR AC ---
        # Following the TODOs below to implement critic loss
        # with torch.no_grad():

            
            # # TODO: Generate noise and clipped the noise.
            # # Hint: Sample a noise from Normal distribution with the scale self.policy_noise.
            # # noise = (
            # #         ???
            # # ).clamp(-self.noise_clip, self.noise_clip)
            # # pass
            # noise = (
            #         torch.randn_like(action)*self.policy_noise
            # ).clamp(-self.noise_clip, self.noise_clip)

            # # TODO: Select next action according to the delayed-updated policy (self.actor_target) and add noise.
            # # next_action = (
            # #         ???
            # # ).clamp(-self.max_action, self.max_action)
            # # pass
            # next_action = (
            #         self.actor_target(next_state)+noise
            # ).clamp(-self.max_action, self.max_action)

            # # TODO: Compute the target Q value (the objective of both critics).
            # # Hint: Call the delayed-updated critic (self.critic_target) first, then compute the critic objective.
            # tq1 = self.critic_target(next_state,next_action)
            # target_Q = reward + self.discount*not_done*tq1
            # # pass
        #----DO NOT USE FOR AC ---
        # target_Q = self.critic_target(next_state, self.actor_target(next_state))
        with torch.no_grad():
            target_Q = self.criticLL_target(self.critic_target(next_state, self.actorLL_target(self.actor_target(next_state))))
            target_Q = reward + (not_done * self.discount * target_Q)

        # Get current Q estimates
        # current_Q1 = self.critic(state, action)
        logits = self.critic(state, action)
        current_Q1 = self.criticLL(logits)

        # TODO: Compute critic loss.
        # Hint: Compute the MSE for both critics and add them up.
        # critic_loss = None
        # pass
        criterion = nn.MSELoss(reduction="none")
        critic_loss = criterion(current_Q1,target_Q)

        minibatchWeightTensor = torch.from_numpy(minibatchWeight).unsqueeze(1).to(device)

        critic_loss = critic_loss*minibatchWeightTensor
        critic_loss = critic_loss.mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        self.criticLL_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.criticLL_optimizer.step()

        actor_loss = None

        # Delayed policy updates
        # if self.total_it % self.policy_freq == 0:
        # TODO: Compute actor loss
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) # regular update for actor
        # actor_loss = -self.critic(state,self.actor(state)).mean()
        # actor_loss = (-self.criticLL(self.critic(state,self.actorLL(self.actor(state))))*minibatchWeightTensor).mean()
        logits = self.critic(state,self.actorLL(self.actor(state)))
        actor_loss = -self.criticLL(logits) * minibatchWeightTensor
        actor_loss = actor_loss.mean()
        
        # pass
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        self.actorLL_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actorLL_optimizer.step()

        # Update the target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.criticLL.parameters(), self.criticLL_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actorLL.parameters(), self.actorLL_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": np.nan if actor_loss is None else actor_loss.item(),
            "critic_loss": np.nan if critic_loss is None else critic_loss.item(),
            "target_q": target_Q.mean().item(),
            "current_q1": current_Q1.mean().item(),
            "reward": reward.mean().item(),
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), os.path.join(filename, "critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(filename, "actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(filename, "actor_optimizer"))

        torch.save(self.criticLL.state_dict(), os.path.join(filename, "criticLL"))
        torch.save(self.criticLL_optimizer.state_dict(), os.path.join(filename, "criticLL_optimizer"))

        torch.save(self.actorLL.state_dict(), os.path.join(filename, "actorLL"))
        torch.save(self.actorLL_optimizer.state_dict(), os.path.join(filename, "actorLL_optimizer"))

    def load(self, filename):
        self.critic.load_state_dict(torch.load(os.path.join(filename, "critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename, "critic_optimizer")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(filename, "actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(filename, "actor_optimizer")))
        self.actor_target = copy.deepcopy(self.actor)

        self.criticLL.load_state_dict(torch.load(os.path.join(filename, "criticLL")))
        self.criticLL_optimizer.load_state_dict(torch.load(os.path.join(filename, "criticLL_optimizer")))
        self.criticLL_target = copy.deepcopy(self.criticLL)

        self.actorLL.load_state_dict(torch.load(os.path.join(filename, "actorLL")))
        self.actorLL_optimizer.load_state_dict(torch.load(os.path.join(filename, "actorLL_optimizer")))
        self.actorLL_target = copy.deepcopy(self.actorLL)
