# generate_synthetic.py
"""Fast synthetic dataset generator for voxel path‑planning.

Key speed optimisations
-----------------------
1. **Geometry optional** – set `CREATE_GEOMETRY = False` (default) to skip
   every `primitive_cube_add` call and run Blender purely as a Python host.
   Runtime drops from tens of seconds per scene to well under a second.
2. **Single‑pass occupancy filling** – boundary walls are written directly to
   the NumPy array; we no longer duplicate loops.
3. **Scene clearing avoided** when no geometry is generated.

If you *do* want visible voxels, flip `CREATE_GEOMETRY = True`; all the old
behaviour remains, but grouped behind a single flag.
"""

import os
import random
import json
from heapq import heappush, heappop

import numpy as np

# --------------------------------------------------
# ▶️  CONFIG
# --------------------------------------------------
OUTPUT_DIR = r".\sample_dataset"  # absolute path on Windows
N_VOXEL = 32                      # grid size (32×32×32)
N_SAMPLES = 10                    # how many scenes to generate
OBSTACLE_DENSITY = 0.05           # fraction of interior voxels to fill
CREATE_GEOMETRY = False           # keep False for max speed

# ‑‑‑ Conditional Blender imports & helpers ‑‑‑
if CREATE_GEOMETRY:
    import bpy
    
    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def add_cube(location, size=1.0, name="Voxel"):
        bpy.ops.mesh.primitive_cube_add(size=size, location=location)
        obj = bpy.context.active_object
        obj.name = name
        return obj
else:
    def clear_scene():
        """No‑op when geometry generation disabled."""
        return

    def add_cube(*args, **kwargs):
        """Stub so the core code doesn’t change."""
        return None

# --------------------------------------------------
# 🔀  ACTION MAP & PATH UTILS
# --------------------------------------------------
ACTION_OFFSETS = {
    0: ( 0,  0,  1),  # FORWARD (+Z)
    1: ( 0,  0, -1),  # BACK    (-Z)
    2: (-1,  0,  0),  # LEFT    (-X)
    3: ( 1,  0,  0),  # RIGHT   (+X)
    4: ( 0,  1,  0),  # UP      (+Y)
    5: ( 0, -1,  0),  # DOWN    (-Y)
}
IDX2ACTION = {i: a for i, a in enumerate(
    ['FORWARD', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN'])}


def astar(occ, start, goal):
    """3‑D Manhattan‑heuristic A* over a binary occupancy volume."""
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    open_set = []
    heappush(open_set, (h(start, goal), 0, start, [start]))
    visited = {start}

    while open_set:
        f, g, current, path = heappop(open_set)
        if current == goal:
            return path
        for dx, dy, dz in ACTION_OFFSETS.values():
            nb = (current[0] + dx, current[1] + dy, current[2] + dz)
            if (
                0 <= nb[0] < N_VOXEL and 0 <= nb[1] < N_VOXEL and 0 <= nb[2] < N_VOXEL and
                occ[nb] == 0 and nb not in visited
            ):
                visited.add(nb)
                heappush(open_set, (g + 1 + h(nb, goal), g + 1, nb, path + [nb]))
    return None  # no path


def path_to_actions(path):
    actions = []
    for a, b in zip(path, path[1:]):
        delta = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        for idx, off in ACTION_OFFSETS.items():
            if off == delta:
                actions.append(idx)
                break
    return actions


def count_turns(actions):
    return sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])

# --------------------------------------------------
# 🏗️  WORLD GENERATION HELPERS
# --------------------------------------------------

def set_boundaries(occ):
    """Mark outermost voxels as occupied (walls)."""
    occ[0, :, :] = 1; occ[-1, :, :] = 1
    occ[:, 0, :] = 1; occ[:, -1, :] = 1
    occ[:, :, 0] = 1; occ[:, :, -1] = 1
    if CREATE_GEOMETRY:
        # Geometry only once per wall voxel set
        for x in range(N_VOXEL):
            for y in range(N_VOXEL):
                add_cube((x, y, 0), name="Wall")
                add_cube((x, y, N_VOXEL - 1), name="Wall")
        for x in range(N_VOXEL):
            for z in range(N_VOXEL):
                add_cube((x, 0, z), name="Wall")
                add_cube((x, N_VOXEL - 1, z), name="Wall")
        for y in range(N_VOXEL):
            for z in range(N_VOXEL):
                add_cube((0, y, z), name="Wall")
                add_cube((N_VOXEL - 1, y, z), name="Wall")


def scatter_obstacles(occ):
    n_interior = (N_VOXEL - 2) ** 3
    n_to_place = int(n_interior * OBSTACLE_DENSITY)
    placed = 0
    while placed < n_to_place:
        x = random.randint(1, N_VOXEL - 2)
        y = random.randint(1, N_VOXEL - 2)
        z = random.randint(1, N_VOXEL - 2)
        if occ[x, y, z] == 0:
            occ[x, y, z] = 1
            if CREATE_GEOMETRY:
                add_cube((x, y, z), name="Obstacle")
            placed += 1


def random_free_cell(occ):
    while True:
        x = random.randint(1, N_VOXEL - 2)
        y = random.randint(1, N_VOXEL - 2)
        z = random.randint(1, N_VOXEL - 2)
        if occ[x, y, z] == 0:
            return (x, y, z)

# --------------------------------------------------
# 🚀  MAIN
# --------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sample_idx in range(N_SAMPLES):
        clear_scene()
        occupancy = np.zeros((N_VOXEL, N_VOXEL, N_VOXEL), dtype=np.uint8)

        # 1️⃣  Walls & obstacles
        set_boundaries(occupancy)
        scatter_obstacles(occupancy)

        # 2️⃣  Start / goal
        start = random_free_cell(occupancy)
        goal = random_free_cell(occupancy)
        # Ensure free
        occupancy[start] = 0
        occupancy[goal] = 0

        # 3️⃣  Pathfinding
        path = astar(occupancy, start, goal)
        if path is None:
            print(f"⏩  Sample {sample_idx}: no path found, skipping")
            continue

        actions = path_to_actions(path)
        n_turns = count_turns(actions)

        # 4️⃣  Build multi‑channel voxel tensor
        ch_obs = occupancy.astype(np.float32)
        ch_start = np.zeros_like(ch_obs); ch_start[start] = 1.0
        ch_goal = np.zeros_like(ch_obs); ch_goal[goal] = 1.0
        voxel_in = np.stack([ch_obs, ch_start, ch_goal], axis=0)

        # 5️⃣  Save
        fname = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:04d}.npz")
        np.savez_compressed(
            fname,
            voxel_data=voxel_in,
            positions=np.array([start, goal], dtype=np.int64),
            target_actions=np.array(actions, dtype=np.int64),
            num_turns=n_turns,
            path_length=len(actions),
        )
        print(f"💾  Saved sample {sample_idx} → {fname}")

    print("✅  Dataset generation complete.")
