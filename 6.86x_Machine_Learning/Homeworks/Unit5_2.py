# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:51:46 2020

@author: user
"""

def T(s0,a,s1):
    """
    
    Parameters
    ----------
    s0 : int
        current state: number between 1 and 5
    s1 : int
        new state after transition
    a : int
        action: +1 - move right, -1 - move left, 0 - stay

    Returns
    -------
    Transition probability form s0 to s1 if action a is taken
    """
    if a == 0:
        if s0 == s1:
            return 0.5
        elif s0 == 1 and s1 == 2:
            return 0.5
        elif s0 == 5 and s1 == 4:
            return 0.5
        elif s0 in [2, 3, 4] and abs(s1 - s0) == 1:
            return 0.25
        else:
            return 0
    else:
        if s0 + a in [1, 2, 3, 4, 5] and s0 == s1:
            return 2/3
        elif s0 + a in [1, 2, 3, 4, 5] and s1 == s0 + a:
            return 1/3
        elif s0 + a == 0 and s1 in [1, 2]:
            return 0.5
        elif s0 + a == 6 and s1 in [4, 5]:
            return 0.5
        else:
            return 0
        
def R(s):
    if s == 5:
        return 1
    else:
        return 0
    
gamma = 0.5

# Initialization`1
V = dict()
for s in range(1, 6):
    V[s] = 0
    
for _ in range(10):
    for s0 in range(1, 6):
        V[s0] = max([sum([T(s0, a, s1) * (R(s1) + gamma * V[s1]) for s1 in range(1, 6)])
                                                                for a in [-1, 0, 1]])
print(V)
            
            