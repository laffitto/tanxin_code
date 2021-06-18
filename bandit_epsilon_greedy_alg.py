import numpy as np 
import matplotlib.pyplot as plt

class Bandit: 
    def __init__(self): 
        self.arm_values = np.random.normal(0,1,10) 
        self.K = np.zeros(10) 
        self.est_values = np.zeros(10) 

    def get_reward(self,action): 
        noise = np.random.normal(0,0.1)
        reward = self.arm_values[action] + noise
        # reward = self.arm_values[action]
        return reward 

    def choose_eps_greedy(self,epsilon):
        rand_num = np.random.random() 
        if epsilon>rand_num: 
            return np.random.randint(10) 
        else: 
            return np.argmax(self.est_values)

    def update_est(self,action,reward): 
        self.K[action] += 1 
        alpha = 1./self.K[action]
        # keeps running average of rewards
        self.est_values[action] += alpha * (reward - self.est_values[action])

def experiment(bandit,Npulls,epsilon):
    step_reward = []
    avgacc_reward = [0]     # average accumulated reward
    for i in range(Npulls): 
        action = bandit.choose_eps_greedy(epsilon)
        R = bandit.get_reward(action) 
        bandit.update_est(action,R) 
        step_reward.append(R)
        avgacc_reward.append((i*avgacc_reward[-1]+R)/(i+1))
    return np.array(step_reward), np.array(avgacc_reward[1:])

Nexp = 20
Npulls = 300
avg_outcome_eps0p0 = np.zeros(Npulls) 
avg_outcome_eps0p01 = np.zeros(Npulls) 
avg_outcome_eps0p1 = np.zeros(Npulls) 
avg_avgacc_eps0p0 = np.zeros(Npulls)
avg_avgacc_eps0p01 = np.zeros(Npulls)
avg_avgacc_eps0p1 = np.zeros(Npulls)

for i in range(Nexp): 
   bandit = Bandit()
   [step_reward, avgacc_reward]= experiment(bandit,Npulls,0.0)
   avg_outcome_eps0p0 += step_reward
   avg_avgacc_eps0p0 += avgacc_reward

   bandit = Bandit() 
   [step_reward, avgacc_reward]= experiment(bandit,Npulls,0.01)
   avg_outcome_eps0p01 += step_reward
   avg_avgacc_eps0p01 += avgacc_reward

   bandit = Bandit() 
   [step_reward, avgacc_reward]= experiment(bandit,Npulls,0.1)
   avg_outcome_eps0p1 += step_reward
   avg_avgacc_eps0p1 += avgacc_reward

avg_outcome_eps0p0 /= np.float(Nexp) 
avg_outcome_eps0p01 /= np.float(Nexp)
avg_outcome_eps0p1 /= np.float(Nexp)
avg_avgacc_eps0p0 /= np.float(Nexp)
avg_avgacc_eps0p01 /= np.float(Nexp)
avg_avgacc_eps0p1 /= np.float(Nexp)

plt.plot(avg_outcome_eps0p0,label="outcome, eps = 0.0")
plt.plot(avg_outcome_eps0p01,label="outcome, eps = 0.01")
plt.plot(avg_outcome_eps0p1,label="outcome, eps = 0.1")
plt.plot(avg_avgacc_eps0p0,label="avgacc, eps = 0.0")
plt.plot(avg_avgacc_eps0p01,label="avgacc, eps = 0.01")
plt.plot(avg_avgacc_eps0p1,label="avgacc, eps = 0.1")
plt.ylim(0,2.5)
plt.legend() 
plt.show()