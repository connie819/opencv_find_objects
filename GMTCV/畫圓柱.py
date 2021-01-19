import numpy as np
import matplotlib.pyplot as plt
from math import factorial
obstacle=[[159,12,60],
           [2,2,2]]
print(obstacle[0][1])
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum( comb(n, i) * t**i * (1 - t)**(n - i) * points[i] for i in range(n + 1) )
def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:,0], new_points[:,1],new_points[:,2]
points = np.array([ [58, -187, -40], [150, 65,-12], [241, -72,19], [196, 120,-63] ])
fig = plt.figure()
ax = fig.gca(projection='3d')
x, y ,z = points[:,0], points[:,1] ,points[:,2]
bx, by ,bz= evaluate_bezier(points, 50)
ax.plot(bx, by, bz, color='gray', label='Arm path')
plt.show()

