import numpy as np
import random

Gamma = 0.8
max_iterations = 1000
R = np.array([
[-1,-1,-1,-1, 0,-1 ],
[-1,-1,-1, 0,-1,100],
[-1,-1,-1, 0,-1,-1 ],
[-1, 0, 0,-1, 0,-1 ],
[ 0,-1,-1, 0,-1,100],
[-1, 0,-1,-1, 0,100],
])
Q = np.array([
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
])

def getQmax(s):
    Qm = -1
    for i in range(0,len(Q[s])):
        if Q[s,i] >= Qm:
            Qm = Q[s,i]

    return Qm

def Q_func(s,a):
    Q[s,a] = R[s,a] + Gamma*getQmax(a)

def get_random_a_according_to_s(s):
    line = R[s]
    r = list(range(0,len(line)))
    random.shuffle(r)
    for i in range(0,len(r)):
        if(R[s,r[i]] >= 0):
            return r[i]

def get_next_state(s,a):
    return a
# algorithm start from here

init_s = int(np.floor(random.random()*6)) # 0~6
print('init s:',init_s)
i=0
s = init_s

while(i<max_iterations):
    a = get_random_a_according_to_s(s)
    Q_func(s,a)
    next_state = get_next_state(s,a)
    s = next_state
    i+=1
    print('iteration: ',i)

print(R)
print(Q)

print("Test")
