import scNodes.core.particlefitting as pfit
from scNodes.core.datatypes import Frame
from skimage.feature import peak_local_max
import numpy as np

f = Frame("C:/Users/Mart/Desktop/TEST_IMG.tif")
f.load()
f.maxima = peak_local_max(f.load(), threshold_abs=300)

print(f.maxima)
print(f.maxima.shape)

particles = pfit.frame_to_particles(f)

for p in particles:
    print(p)