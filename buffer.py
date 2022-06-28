from collections import deque
import random
from utilities import transpose_list


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        input_to_buffer = transpose_list(transition)
        #print("len(input_to_buffer) = ", len(input_to_buffer)) # 1
    
        for item in input_to_buffer:
            #print(item[0].shape) # (2, 24)
            #print(item[1].shape) # (48,)
            #print(item[2].shape) # (2, 2)
            #print(item[3].shape) # (2,)
            #print(item[4].shape) # (2, 24)
            #print(item[5].shape) # (48,)
            #print(item[6].shape) # (2,)
            self.deque.append(item) # appends a list of (s,a,r,s') for each environment. 

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)



