import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Parameters
NX = 40
NY = 20
H_inlet = NY/2
H_outlet = NY
x_expansion = NX/2

# Node types
INLET, OUTLET, SOUTH, NORTH, EAST, BULK, SOLID, CORNER1, CORNER2 = range(9)
labels = ["INLET", "OUTLET", "SOUTH", "NORTH", "EAST", "BULK", "SOLID", "CORNER1", "CORNER2"]
colors = ["green", "red", "black", "blue", "orange", "lightblue", "gray", "purple", "brown"]

# Grid classification
grid = np.full((NX, H_outlet), BULK, dtype=int)

for x in range(NX):
    for y in range(H_outlet):

        yy = (NY-1)-y
        if yy == 0:
            grid[x, y] = SOUTH
        elif x == x_expansion-1 and yy == H_inlet - 1:
            grid[x, y] = CORNER1
        elif x == x_expansion-1 and yy == H_outlet - 1:
            grid[x, y] = CORNER2

        elif x < x_expansion and yy == H_inlet - 1:
            grid[x, y] = NORTH

        elif x >= x_expansion and yy == H_outlet - 1:
            grid[x, y] = NORTH

        elif x < x_expansion-1 and H_inlet <= yy < H_outlet:
            grid[x, y] = SOLID
        elif x == x_expansion-1 and H_inlet <= yy < H_outlet:
            grid[x, y] = EAST

        # --- Inlet and outlet (only if not overwritten by walls) ---
        elif x == 0 and yy < H_inlet:
            grid[x, y] = INLET
        elif x == NX - 1 and yy < H_outlet:
            grid[x, y] = OUTLET

# Plot
cmap = mcolors.ListedColormap(colors)
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(grid.T, origin="lower", cmap=cmap, aspect="equal")

# Legend
legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=colors[i], markersize=10,
                              label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
