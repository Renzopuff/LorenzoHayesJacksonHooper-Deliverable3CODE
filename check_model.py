from train import train_once
print([round(train_once(True, sigma=0.0001, seed=s), 3) for s in range(3)])