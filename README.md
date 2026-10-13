# PINN Mechanics Mini-Project

## Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network

This project implements three Physics-Informed Neural Network (PINN) experiments based on the assigned paper:

1. **1D consolidation** — diffusion PDE with an analytical reference solution.
2. **2D steady-state heat conduction** — Poisson/heat equation.
3. **Inverse structural dynamics** — identification of damping `c` and stiffness `k`.

### Repository contents

- `notebooks/PINN_Mechanics_Final_Local.ipynb` — final executable notebook.
- `report/PINN_Mechanics_Report.pdf` — two-page project write-up.
- `requirements.txt` — Python dependencies.

### How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook in Jupyter Notebook/JupyterLab and use **Run All**.

The notebook defaults to:

```python
RUN_MODE = "validation"
```

This mode is intentionally lightweight so the complete demonstration can run locally on a normal CPU. The notebook also contains:

```python
RUN_MODE = "paper"
```

which switches to the paper-reported training configuration. Paper mode is computationally expensive and is not required for the live demonstration.

### Method

The PINN approximates the physical field with a neural network. Automatic differentiation is used to construct governing-equation residuals. The training objective combines physics, boundary/initial-condition, and data losses:

`L = wf Lf + wb Lb + wd Ld`

For the inverse dynamics experiment, the unknown physical parameters `c` and `k` are optimized together with the network parameters.

### Validation results

The reduced validation run is a practical demonstration and **not an exact numerical reproduction of the paper's final reported values**.

- Experiment 1 MSE: `0.005721172793339642`
- Experiment 1 maximum absolute error: `0.9098457509015547`
- Experiment 2 PDE residual MSE: `0.013626818545162678`
- Experiment 2 boundary error: `0.01187849`
- Experiment 3 identified damping `c`: `1.105580` (true value 0.4)
- Experiment 3 identified stiffness `k`: `1.047393` (true value 4.0)

### Reproducibility note

The paper's reported architectures, loss formulation, point counts, learning rates, and long training schedules are retained in the notebook's paper mode. The observation interval for Experiment 3 is not stated in the extracted paper text, so this implementation explicitly uses `[0, 10]`.

