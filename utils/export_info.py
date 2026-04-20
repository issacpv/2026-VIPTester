import numpy as np
import pybullet as p

 

def export_obj_bottom(file_path, bricks, local_bottom_vertices):

    """
    Export OBJ file drawing planar bottom face in the format:
    # n vertices m faces
    v x y z
    v x y z
    ...
    f v1 v2 v3 ...
    ...
    No object/group lines which is different from standard OBJ files.
    """
    vertices = []
    faces = []

    # Build global vertex and face lists
    for i, b in enumerate(bricks):
        pos, orn = p.getBasePositionAndOrientation(b)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        O = np.asarray(pos, dtype=float)

        bottom_vertices_per_tile = local_bottom_vertices[i]
        if not bottom_vertices_per_tile or len(bottom_vertices_per_tile) < 3:
            continue

        base = len(vertices) + 1  # OBJ is 1-based
        # transform and append vertices
        for v in bottom_vertices_per_tile:
            vw = R @ np.asarray(v) + O
            vertices.append(vw)

        n = len(bottom_vertices_per_tile)
        faces.append(list(range(base, base + n)))

    # Write file: header, then v, then f
    with open(file_path, "w") as obj:
        # include header
        obj.write(f"# {len(vertices)} vertices {len(faces)} faces\n")
        
        for v in vertices:
            obj.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        obj.write("\n")
        
        for f in faces:
            obj.write("f " + " ".join(map(str, f)) + "\n")

    print(f"Wrote planar OBJ: {file_path}")