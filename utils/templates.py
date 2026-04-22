# utils/templates.py
import numpy as np

# --- Tetrahedron ---
TET_VERTS = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
], dtype=float) / np.sqrt(3)   # unit circumradius

TET_FACES = [[0,1,2], [0,2,3], [0,3,1], [1,3,2]]

# --- Hex prism ---
# vertex order: 0..5 bottom CCW from above, 6..11 top matching
HEX_PRISM_FACES = [
    [0,1,2,3,4,5],          # bottom
    [11,10,9,8,7,6],        # top
    [0,6,7,1], [1,7,8,2], [2,8,9,3],
    [3,9,10,4], [4,10,11,5], [5,11,6,0],
]
HEX_PRISM_ENDCAP_BOTTOM = [0,1,2,3,4,5]
HEX_PRISM_ENDCAP_TOP    = [6,7,8,9,10,11]

# --- Oct prism --- (similar structure)

# --- TCO hub ---
TCO_VERTS = np.array([...])            # 48 canonical local positions
TCO_FACES = [...]                      # 26 face lists
TCO_OCT_FACES = [f0, f1, f2, f3, f4, f5]   # indices into TCO_FACES
TCO_HEX_FACES = [f6, f7, ..., f13]
TCO_SQ_FACES  = [f14, ..., f25]

TEMPLATES = {
    'tet':       {'verts': TET_VERTS,       'faces': TET_FACES},
    'hex_prism': {'verts': None,            'faces': HEX_PRISM_FACES},   # verts come from input
    'oct_prism': {...},
    'tco_hub':   {'verts': TCO_VERTS,       'faces': TCO_FACES,
                  'groups': {'oct': TCO_OCT_FACES, 'hex': TCO_HEX_FACES, 'sq': TCO_SQ_FACES}},
}