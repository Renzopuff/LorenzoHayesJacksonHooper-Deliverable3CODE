import time
from train import train_once
t = time.time()
print(train_once(True, sigma=1.0, seed=0), "score;", time.time() - t, "seconds")