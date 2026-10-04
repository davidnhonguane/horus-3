"""Horus - forest risk & emergency intelligence on top of drone LiDAR / multispectral data.

Modules
-------
geo       coordinate handling (ETRS-TM35FIN, local tangent plane, haversine)
scan3d    read any 3D scan: LAS/LAZ, PLY, OBJ, GLB, XYZ/TXT/CSV/PTS, E57
lasio     dependency-free LAS 1.2-1.4 reader/writer (LAZ via optional laspy)
synth     deterministic synthetic forest estate + DJI-L3-like point cloud
lidar     ground filter, DTM / DSM / CHM, individual tree detection, species, health
weather   live weather (Open-Meteo) + scenarios + Canadian Fire Weather Index
fire      fuel mapping (FBP fuel types), static fire-risk index, fire-spread simulation
wind      windthrow probability per tree, strike probability on assets, road blockage
routing   vehicle-class routing on terrain/vegetation cost surface w/ blockages & fire
dispatch  drone-operator scheduling on Forey's dataset + storm emergency re-planning
site      orchestrates the pipeline for one surveyed area and caches results
server    zero-dependency HTTP API + web UI
"""

__version__ = "1.5.0"   # 1.4: automatic storm + fire simulation on every 3D scan (shown in 3D), NEW roads / power lines always highlighted; 1.3: photo analysis (drone / aerial / ground photos), auto-start uploads, STL/glTF; 1.2: any 3D scan, tree hazards, emergency plan
