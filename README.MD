
# 🚁 🛸 **KQ-LMPC: Koopman LMPC for Quadrotors**

#### **Analytical Koopman Lifting → Convex MPC (< 10ms Jetson ![Jetson Optimized](https://img.shields.io/badge/Jetson-NVIDIA-green?logo=nvidia) ) → Real-Time Flight on SE(3)**  

<h4> <code>kq_lmpc_quadrotor</code></b>: A complete, hardware-ready Python package for <b>Koopman-based Linear Model Predictive Control (LMPC)</b>, delivering <b>real-time trajectory tracking</b> for quadrotors using <b>analytical Koopman lifting</b> (no training data required). </h4>

[![PyPI Version](https://img.shields.io/pypi/v/kq-lmpc-quadrotor.svg)](https://pypi.org/project/kq-lmpc-quadrotor/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Paper](https://img.shields.io/badge/IEEE-RA--L%202025-red)](https://www.researchgate.net/publication/396545942_Real-Time_Linear_MPC_for_Quadrotors_on_SE3_An_Analytical_Koopman-based_Realization)
[![Stars](https://img.shields.io/github/stars/santoshrajkumar/kq-lmpc-quadrotor?style=social)]()

<p align="center">
  ✅ First hardware-deployed real-time Linear MPC for quadrotors derived using Koopman operator theory, no data required.
</p>

<div align="center">
🔧 **acados-powered QP** <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20" />
 | 🧭 **LPV + LTI Koopman Embedding** |  🚀 **PX4/ROS2/MAVSDK Ready**

`pip install kq-lmpc-quadrotor`

> **Zero Data • Zero Neural Networks • Fully Explainable Control**

<table align="center">
  <tr>
    <td>
      <img src="media/dg.GIF" height="200px" width="350px" style="border-radius:8px;"/>
    </td>
    <td>
      <img src="media/dg2.gif" height="200px" width="350px" style="border-radius:8px;"/>
    </td>
  </tr>
</table>


<p align="center">
  <img src="https://img.shields.io/badge/Koopman-Lifting-blue" />
  <img src="https://img.shields.io/badge/KQ--LMPC-Convex%20MPC-red" />
  <img src="https://img.shields.io/badge/Python-Package-success" />
  <img src="https://img.shields.io/badge/Research-Backed-lightgrey" />
</p>
</div>
---


## 🌟 Key Features

✅ **Analytical Koopman lifting** with **generalizable observables**  
&nbsp;&nbsp;&nbsp;&nbsp;→ No neural networks, no training, no data fitting required  

✅ **Data-free Koopman-lifted LTI + LPV models**  
&nbsp;&nbsp;&nbsp;&nbsp;→ Derived directly from **SE(3) quadrotor dynamics** using Lie algebra structure  

✅ **Real-time Linear MPC (LMPC)**  
&nbsp;&nbsp;&nbsp;&nbsp;→ Solved as a **single convex QP** termed **KQ-LMPC**  
&nbsp;&nbsp;&nbsp;&nbsp;→ < **10 ms** solve time on **Jetson NX / embedded hardware**  

✅ **Trajectory tracking on SE(3)**  
&nbsp;&nbsp;&nbsp;&nbsp;→ **Provable controllability** in lifted Koopman space  

✅ **Closed-loop robustness guarantees**  
&nbsp;&nbsp;&nbsp;&nbsp;→ **Input-to-state practical stability (I-ISpS)**  

✅ **Hardware-ready integration**  
&nbsp;&nbsp;&nbsp;&nbsp;→ Works with **PX4 Offboard Mode**, **ROS2**, **MAVSDK**, **MAVROS**  

✅ **Drop-in MPC module**  
&nbsp;&nbsp;&nbsp;&nbsp;→ for both KQ-LMPC, NMPC with acados on Python.

---

## 🧠 Paper

This work is based on:

> **"Real-Time Linear MPC for Quadrotors on SE(3): An Analytical Koopman-based Realization"**  
> *IEEE Robotics and Automation Letters (RA-L), 2025* (To appear)  
> **Santosh Rajkumar**, Chengyu Yang, Yuliang Gu, Sheng Cheng, Naira Hovakimyan, Debdipta Goswami  
> [[Paper PDF]](media/paper.pdf) • [[ArXiv]](https://arxiv.org/abs/2409.12374) • [[Video Demos]](https://soarpapers.github.io/)

If you use this repository, **please cite us** 🙏

```bibtex
@article{rajkumar2025kqlmpc,
  title={Real-Time Linear MPC for Quadrotors on SE(3): An Analytical Koopman-based Realization},
  author={Rajkumar, Santosh and Yang, Chengyu and Gu, Yuliang and Cheng, Sheng and Hovakimyan, Naira and Goswami, Debdipta},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```
## 🔧 Installation

*Virtual environment recommended

Install from PyPI (recommended):

```bash
pip install kq_lmpc_quadrotor
```

Install from source

```bash
git clone https://github.com/santoshrajkumar/kq-lmpc-quadrotor.git
cd kq-lmpc-quadrotor
pip install -e .
```

## ⚡ Quick Demo
```bash
from kq_lmpc_quadrotor import lqr_demo
lqr_demo()
```
> ⚠️ **Important Dependency Notice**
> This package relies on **acados** for fast Model Predictive Control (MPC).  
> You must configure **acados + Python interface** before running MPC examples.
> Recommended that you install acados and configure, before installing the package (for MPC)
> Koopman Lifting and LQR will work without acados

> **Quick Setup Checklist**
> - Install acados ✅
> - Enable Python interface ✅
> - Export `ACADOS_SOURCE_DIR` ✅
> - Set library paths:
>   - Linux: `LD_LIBRARY_PATH`
>   - macOS: `DYLD_LIBRARY_PATH`
>
> 📚 Install acados: https://docs.acados.org/installation/index.html  
> 🐍 acados Python Interface: https://docs.acados.org/python_interface/index.html  
> 💻 OS Support: **Linux/macOS** (*Not tested on Windows*)

```bash
from kq_lmpc_quadrotor import kqlmpc_demo
kqlmpc_demo()
```

### 📊 Benchmarking: KQ-LMPC vs NMPC (Numerical Simulation)

**Experimental Setup:**  
Python 3.10 • Ubuntu 22.04 • AMD Ryzen 3 PRO CPU  
Metrics: **Mean solve time** (μₜ), **Worst-case solve time** (tᵥ), and **Tracking RMSE** (𝓔ₛ).  
Prediction horizon **Tₕ ∈ {0.8, 1.4, 2.0, 2.8} s** across **4 tasks**.

---

| **Method** | **Metric** | **Tₕ = 0.8 s** (Tasks 1–4) | **Tₕ = 1.4 s** (Tasks 1–4) | **Tₕ = 2.0 s** (Tasks 1–4) | **Tₕ = 2.8 s** (Tasks 1–4) |
|------------|------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| **KQ-LMPC** | **μₜ (ms)** | 0.32 • 0.32 • 0.33 • 0.34 | 0.47 • 0.47 • 0.51 • 0.50 | 0.78 • 0.80 • 0.90 • 0.87 | 1.04 • 1.07 • 1.31 • 1.23 |
|            | **tᵥ (ms)** | 0.79 • 0.76 • 1.00 • 0.98 | 1.28 • 1.26 • 1.48 • 1.26 | 2.25 • 2.18 • 2.49 • 2.13 | 2.63 • 2.85 • 5.00 • 3.32 |
|            | **𝓔ₛ (m)**  | 0.06 • 0.09 • 0.10 • 0.13 | 0.05 • 0.06 • 0.14 • 0.18 | 0.05 • 0.04 • 0.10 • 0.12 | 0.05 • 0.05 • 0.14 • 0.15 |
| **NMPC**    | **μₜ (ms)** | 0.86 • 0.97 • 1.18 • 1.46 | 1.14 • 1.20 • 1.68 • 2.05 | 1.69 • 1.75 • 2.70 • 3.24 | 1.96 • 2.13 • 3.35 • 4.15 |
|            | **tᵥ (ms)** | 2.38 • 2.20 • 2.46 • 3.88 | 3.07 • 4.12 • 5.25 • 6.48 | 4.48 • 4.53 • 8.52 • 9.47 | 4.66 • 6.60 • 10.68 • 11.78 |
|            | **𝓔ₛ (m)**  | 0.05 • 0.07 • 0.09 • 0.09 | 0.06 • 0.06 • 0.10 • 0.12 | 0.04 • 0.06 • 0.06 • 0.07 | 0.04 • 0.05 • 0.08 • 0.09 |

---

### 🔍 Highlights

- ✅ **2–4× faster mean computation time** than NMPC  
- ✅ **Lower worst-case latency** → more reliable for real-time flight  
- ✅ **Competitive tracking accuracy**  
- ✅ **Scales efficiently** with larger prediction horizons

---

🚀 **v2.0 (Coming Soon)**  
🔧 **Complete PX4 Offboard control pipeline for hardware**  
🔧 **Full Gazebo SITL + PX4 integration demos**    
🔧 **Flight-ready example configs**

If you find this project useful, please ⭐ star the repo and follow — your support drives development!
