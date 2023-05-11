import torch
import numpy as np
from collections import defaultdict

class Record:
    def __init__(self, name, num=10, highest=True) -> None:
        self.name = name
        self.num = num
        self.highest = highest
        self.vals = []
        self.items = []

    def update(self, val, item):
        if len(self.vals) < self.num:
            # Add the item and value, and then sort the list
            self.vals.append(val)
            self.items.append(item)
            self.sort()
        else:
            if self.highest:
                if val > self.vals[-1]:
                    # Replace the smallest value and item, and then sort the list
                    self.vals[-1] = val
                    self.items[-1] = item
                    self.sort()
            else:
                if val < self.vals[-1]:
                    # Replace the largest value and item, and then sort the list
                    self.vals[-1] = val
                    self.items[-1] = item
                    self.sort()

    def sort(self):
        sorted_indices = sorted(range(len(self.vals)), key=lambda i: self.vals[i], reverse=self.highest)
        self.vals = [self.vals[i] for i in sorted_indices]
        self.items = [self.items[i] for i in sorted_indices]
        # combined = list(zip(self.vals, self.items)) # zip returns an iterable object, each element is a tuple
        # combined.sort(reverse=self.highest)
        # self.vals, self.items = map(list, zip(*combined)) # map apply list() to every tuple returned by zip

    def __str__(self) -> str:
        return f"{self.name}: {list(zip(self.items, self.vals))}"

class Metric:
    def __init__(self, name) -> None:
        self.name = name
        self.val = 0
        self.count = 0
    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val += val
        self.count += 1
    def avg(self):
        return self.val / max(1, self.count)
    def __str__(self) -> str:
        return f"{self.name}: {self.avg()} (avg over {self.count})"

if __name__ == '__main__':
    
    # Example usage:
    record = Record("Top Scores", num=5, highest=False)
    scores = [100, 50, 75, 120, 80, 60, 110]
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"]

    for name, score in zip(names, scores):
        record.update(score, name)

    print(record)