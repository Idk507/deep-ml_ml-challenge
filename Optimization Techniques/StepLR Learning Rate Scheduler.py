"""
Write a Python class StepLRScheduler to implement a learning rate scheduler based on the StepLR strategy. Your class should have an init method implemented to initialize with an initial_lr (float), step_size (int), and gamma (float) parameter. It should also have a get_lr(self, epoch) method implemented that returns the current learning rate for a given epoch (int). The learning rate should be decreased by gamma every step_size epochs. The answer should be rounded to 4 decimal places. Only use standard Python.

Example:
Input:
scheduler = StepLRScheduler(initial_lr=0.1, step_size=5, gamma=0.5)
print(scheduler.get_lr(epoch=0))
print(scheduler.get_lr(epoch=4))
print(scheduler.get_lr(epoch=5))
print(scheduler.get_lr(epoch=9))
print(scheduler.get_lr(epoch=10))
Output:
0.1
0.1
0.05
0.05
0.025
Reasoning:
The initial learning rate is 0.1. It stays 0.1 for epochs 0-4. At epoch 5, it decays by 0.5 to 0.05. It stays 0.05 for epochs 5-9. At epoch 10, it decays again to 0.025.

"""
class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        # Initialize initial_lr, step_size, and gamma
        self.initial_lr = initial_lr
        self.step_size = step_size 
        self.gamma = gamma


    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch
        decay_steps = epoch // self.step_size 
        current_lr = self.initial_lr * (self.gamma ** decay_steps)
        return round(current_lr,4)
