import numpy as np
import matplotlib.pyplot as plt

def test(x,y,patch_size):
    x = int(x)
    y = int(y)
    position_map = np.zeros((patch_size,patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            d = np.sqrt((i-x)**2 + (j-y)**2)
            # if 1 * np.exp(-d/14) > 0.9:
            #     position_map[i][j] = 1
            position_map[i][j] = 1 * np.exp(-d/10)
    position_map = position_map
    plt.imshow(position_map)
    plt.show()
    print(position_map)
if __name__ == '__main__':
    test(3,3,7)