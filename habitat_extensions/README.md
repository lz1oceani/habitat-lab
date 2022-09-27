## Installation

```bash
conda create -n hab-profile python=3.8 cmake=3.14.0
conda activate hab-profile
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
pip install -r requirements.txt
python setup.py develop
# Generate physical config to correctly configure the simulator backend
# CAUTION: you must modify this config to change simulation timestep!
python -c "from habitat.datasets.utils import check_and_gen_physics_config; check_and_gen_physics_config()"
```

## PickCube

```python
import habitat_extensions.pick_cube
env = gym.make("HabitatPickCube-v0")
```
