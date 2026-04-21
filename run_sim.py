"""
Auxetic structure PyBullet simulation.

The central lattice hub rotates clockwise in the XY plane.
Off-centre ball joints transmit that rotation to each surrounding tetrahedron,
which rotates ~90 degrees relative to the hub as it turns — bringing one of its
faces parallel to the hub face.  Boundary hubs are fixed; everything else is
driven by the mechanism.

Controls (PyBullet window must be in focus):
    Space  – pause / resume
    Q      – quit
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.auxetic_sim import AuxeticSim


def main():
    sim = AuxeticSim()
    sim.build()
    sim.run()


if __name__ == "__main__":
    main()
