# 🚁 `kq_lmpc_quadrotor`: Koopman MPC for Quadrotors Derived From Analytically Derived Koopman Embeddings
> A **complete Python package** for real-time **Koopman-based Linear Model Predictive Control (LMPC)** of quadrotors.  

[![PyPI Version](https://img.shields.io/pypi/v/kq-lmpc.svg)](https://pypi.org/project/kq-lmpc/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Paper](https://img.shields.io/badge/IEEE-RA--L%202025-red)](media/paper.pdf)
[![Stars](https://img.shields.io/github/stars/santoshrajkumar/kq-lmpc-quadrotor?style=social)]()

> **First hardware-deployed real-time Linear MPC for quadrotors derived using Koopman operator theory,  no data required.**  
> Fast QP ⚡ Comparable Performance like NMPC 🎯 Runs on embedded hardware 💻✅

> ✅ **KQ-LMPC = Koopman Lift + Convex MPC + Real-Time Control**
> 
> A unified Python framework:
> - **Koopman lifting** without machine learning
> - **Convex linear MPC** that runs in real time
> - **Provable stability + hardware deployability**

<p align="center">
  <img src="https://img.shields.io/badge/Koopman-Lifting-blue" />
  <img src="https://img.shields.io/badge/KQ--LMPC-Convex%20MPC-red" />
  <img src="https://img.shields.io/badge/Python-Package-success" />
  <img src="https://img.shields.io/badge/Research-Backed-lightgrey" />
</p>

---

<table align="center">
  <tr>
    <td>
      <img src="media/dg.GIF" height="200px" width="400px" style="border-radius:8px;"/>
    </td>
    <td>
      <img src="media/dg2.GIF" height="200px" width="400px" style="border-radius:8px;"/>
    </td>
  </tr>
</table>



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
> *IEEE Robotics and Automation Letters (RA-L), 2025*  
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
lqr_demo
```
```bash
kqlmpc_demo
```
> ⚙️ **Note:** This package uses the acados toolchain for fast MPC.  
> Please ensure that **acados is installed and its Python interface is configured** before running the demo/examples with MPC.  
> Installation guide: https://docs.acados.org/installation/index.html
> Python interface: https://docs.acados.org/python_interface/index.html
> OS requirement: Linux/Mac (Not tested on Windows).
> Ensure that LD_LIBRARY_PATH is set correctly (DYLD_LIBRARY_PATHon MacOS).
> Ensure that ACADOS_SOURCE_DIR is set correctly.

<p align="center">
  <a href="https://colab.research.google.com/github/santoshrajkumar/kq-lmpc-quadrotor/blob/main/notebooks/kq_lmpc_demo.ipynb">
    <img src="https://img.shields.io/badge/Launch%20Demo-Colab-blue?logo=googlecolab&style=for-the-badge" height="40">
  </a>
</p>
