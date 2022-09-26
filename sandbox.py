## Testing .csv export

from dataset import *
import numpy as np

pd = ParticleData()

n = 100
f = np.random.randint(0,  10, n)
x = np.random.uniform(0, 10.0, n)
y = np.random.uniform(0, 10.0, n)
s = np.random.uniform(100.0, 500.0, n)

particles = list()
for i in range(n):
    particles.append(Particle(f[i], x[i], y[i], s[i], 1000.0))

pd += particles
pd.bake()
pd.save_as_csv("test_csv.csv")
