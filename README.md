# deep_spine

**deep_spine** is a Python package for modeling spinal networks and their organization. It leverages the MuJoCo simulator for realistic biomechanical and neuromuscular simulations. The library is designed for researchers and developers working on computational neuroscience, biomechanics, and motor control.

---

## Features

- **Spinal Modeling**: Simulate spinal networks with customizable layers and afferent/efferent dynamics.
- **MuJoCo Integration**: Integration with the MuJoCo physics engine for realistic simulations.
- **Customizable Environments**: Create and modify simulation environments tailored to specific tasks.
- **Data Recording**: Export simulation data and visualizations for analysis.

---

## Installation

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/jonreal/deep-spine.git
   cd deep-spine

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

Install deep_spine:
    ```bash
    pip install .

---

## Run Hebbian Training

To run the Hebbian training example, use the following command:

```bash
cd experiments/hebbian
python train.py
```
