import random
import numpy as np
from itertools import product
from PIL import Image, ImageDraw

def mc_step(n, t, w_a, w_b):
    # find clusters and flip
    visited = np.full((n, t), False)  # record if site has been visited
    x = 0
    y = 0
    clusternr = 0
    # bond assignment
    for x, y in product(range(n), range(t)):
        if y % 2 != x % 2:
            continue
        # all occupied or all unoccupied
        if fermion[x, y] == fermion[(x + 1) % n, y] and fermion[x, (y + 1) % t] == fermion[(x + 1) % n, (y + 1) % t]:
            bond[x // 2, y] = False
        # diagonal occupation
        elif fermion[x, y] != fermion[x, (y + 1) % t]:
            bond[x // 2, y] = True
        # parallel occupation
        else:
            bond[x // 2, y] = False if random.random() < w_a / (w_a + w_b) else True
        # calculate bond config in nicer lattice for debugging purposes
        bond_debug[x, y] = bond[x // 2, y]

    while True:
        while True:
            visited[x, y] = True
            # follow bond loop
            cluster[x, y] = clusternr
            if x % 2 == 0 and y % 2 == 0:
                # top
                if not bond[(x // 2 - 1) % (n // 2), (y - 1) % t] and not visited[x, (y - 1) % t]:
                    y -= 1
                # left
                elif bond[(x // 2 - 1) % (n // 2), (y - 1) % t] and not visited[(x - 1) % n, y]:
                    x -= 1
                # bottom
                elif not bond[x // 2, y] and not visited[x, (y + 1) % t]:
                    y += 1
                # right
                elif bond[x // 2, y] and not visited[(x + 1) % n, y]:
                    x += 1
                # closed loop
                else:
                    break
            elif x % 2 == 1 and y % 2 == 0:
                # top
                if not bond[x // 2, (y - 1) % t] and not visited[x, (y - 1) % t]:
                    y -= 1
                # left
                elif bond[x // 2, y] and not visited[(x - 1) % n, y]:
                    x -= 1
                # bottom
                elif not bond[x // 2, y] and not visited[x, (y + 1) % t]:
                    y += 1
                # right
                elif bond[x // 2, (y - 1) % t] and not visited[(x + 1) % n, y]:
                    x += 1
                # closed loop
                else:
                    break
            elif x % 2 == 0 and y % 2 == 1:
                # top
                if not bond[x // 2, (y - 1) % t] and not visited[x, (y - 1) % t]:
                    y -= 1
                # left
                elif bond[(x // 2 - 1) % (n // 2), y] and not visited[(x - 1) % n, y]:
                    x -= 1
                # bottom
                elif not bond[(x // 2 - 1) % (n // 2), y] and not visited[x, (y + 1) % t]:
                    y += 1
                # right
                elif bond[x // 2, (y - 1) % t] and not visited[(x + 1) % n, y]:
                    x += 1
                # closed loop
                else:
                    break
            elif x % 2 == 1 and y % 2 == 1:
                # top
                if not bond[x // 2, (y - 1) % t] and not visited[x, (y - 1) % t]:
                    y -= 1
                # left
                elif bond[x // 2, (y - 1) % t] and not visited[(x - 1) % n, y]:
                    x -= 1
                # bottom
                elif not bond[x // 2, y] and not visited[x, (y + 1) % t]:
                    y += 1
                # right
                elif bond[x // 2, y] and not visited[(x + 1) % n, y]:
                    x += 1
                # closed loop
                else:
                    break
            x = x % n  # boundary conditions
            y = y % t

        clusternr += 1
        full = False
        for i, j in product(range(n), range(t)):
            if not visited[i, j]:
                x = i
                y = j
                break
            if i == n - 1 and j == t - 1:
                full = True  # if last indices are reached everything has been visited
        if full:
            break




def draw_bonds(n, t):
    scale = 40
    image = Image.new("RGB", (scale*n + 2, scale*t+2), "white")
    draw = ImageDraw.Draw(image)
    for x in range(n):
        for y in range(t):
            if bond_debug[x, y] == 1:
                draw.line([(x*scale, y*scale), ((x+1) * scale, y * scale)], width=scale//10, fill="green", joint="curve")
                draw.line([(x*scale, (y+1)*scale), ((x+1) * scale, (y+1) * scale)], width=scale//10, fill="green", joint="curve")
            elif bond_debug[x,y] == 0:
                draw.line([(x * scale, y * scale), (x * scale, (y+1) * scale)], width=scale // 10, fill="green",
                          joint="curve")
                draw.line([((x+1) * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], width=scale // 10,
                          fill="green", joint="curve")
            np.random.seed(cluster[x, y] + 30)
            color = tuple(np.random.choice(range(256), size=3))
            draw.ellipse((x*scale - 10, y*scale-10, x*scale+10, y*scale+10), fill=color, outline='black')
            if x%2:
                draw.text((x*scale-4, y*scale-4), "+", fill=(0, 0, 0))
            else:
                draw.text((x*scale-4, y*scale-4), "-", fill=(0, 0, 0))




    image.save("config.jpg")


def main():
    global fermion
    global bond
    global bond_debug
    global cluster
    n = 16  # number of lattice points
    t = 8  # number of half timesteps (#even + #odd)
    b = 1   # beta
    mc_steps = 3   # number of mc steps
    initial_mc_steps = 5000
    w_a = 1/2  # np.exp(b/t)  # weight of a plaquettes U = t = 1
    w_b = 1/2  # np.sinh(b/t)  # weight of b plaquettes

    # fermion lattice initialized to reference configuration
    fermion = np.full((n, t), False)
    for i in range(n//2):
        for j in range(t):
            fermion[2*i, j] = True

    cluster = np.full((n, t), -1)

    # bond lattice is squashed down and initalized to vertical plaquettes
    bond = np.full((n//2, t), False)    # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B

    #random.seed(42)
    bond_debug = np.full((n, t), - 1)   # only for debugging purposes


    for mc in range(mc_steps):
        mc_step(n, t, w_a, w_b)

        charge = np.zeros(cluster.max() +1)
        for j in range(t):
            for c in range(cluster.max()+1):
                rowcharge = 0
                for i in range(n):
                    if cluster[i, j] == c:
                        if i % 2:
                            rowcharge += 1
                        else:
                            rowcharge -= 1
                if charge[c] != rowcharge and j > 0:
                    raise('Charge varies over different rows')
                charge[c] = rowcharge
        # print(charge)
        if charge.sum() != 0:
            raise('Total charge not zero')

        if charge.max() > 1:
            if np.count_nonzero(charge == charge.max()) > 1:
                print('multiple 2 windings')
            for c in charge:
                if abs(c) != charge.max() and c != 0:
                    raise('clusters of different charges mixed')
        if charge.max() > 1:
            print(charge.max())

        for j in range(t):
            for i in range(n-2):
                if cluster[i, j] != cluster[i+1, j] and abs(charge[cluster[i, j]]) == abs(charge[cluster[i+1, j]]) == 1:
                    assert(charge[cluster[i, j]] != charge[cluster[i+1, j]])
    draw_bonds(n, t)




if __name__ == "__main__":
    main()