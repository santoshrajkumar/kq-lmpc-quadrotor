
# 🚁 🛸 **KQ-LMPC: Koopman LMPC for Quadrotors**

 > 🚀 **KQ-LMPC is the fastest open-source Koopman MPC controller for quadrotors: zero training data, fully explainable, hardware-proven SE(3) control.** 
 
 > ⚡ Real-time convex MPC (<10 ms) • ✅ Stability guarantees • 🔧 PX4/ROS2 ready

 [![PyPI Version](https://img.shields.io/pypi/v/kq-lmpc-quadrotor.svg)](https://pypi.org/project/kq-lmpc-quadrotor/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Paper](https://img.shields.io/badge/Peer--Reviewed-IEEE%20RA--L%202025-darkred)](https://www.researchgate.net/publication/396545942_Real-Time_Linear_MPC_for_Quadrotors_on_SE3_An_Analytical_Koopman-based_Realization)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17393835.svg)](https://doi.org/10.5281/zenodo.17393835)
[![Stars](https://img.shields.io/github/stars/santoshrajkumar/kq-lmpc-quadrotor?style=social)]()


<h4><code>kq_lmpc_quadrotor</code> — A **hardware-ready Python package** for **Koopman-based Linear Model Predictive Control (LMPC)**. Built for **real-time flight**, powered by **analytical Koopman lifting** (no neural networks, no learning phase).</h4>


<p align="center">
  <a href="https://github.com/santoshrajkumar/kq-lmpc-quadrotor">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo_for_updates!-red?style=for-the-badge">
  </a>
</p>


<p align="center">
  ✅ First hardware-validated Koopman MPC on SE(3)  
  ✅ Convex QP (<10 ms solve time) via acados  
  ✅ PX4/ROS2/MAVSDK compatible : deploy to drones easily  
</p>


<div align="center">

🔧 **acados-powered QP** <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20" />
 | 🧭 **LPV + LTI Koopman Embedding**
 | 🚀 **PX4 Offboard Ready**

</div>


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
  <img src="https://img.shields.io/badge/MATLAB-Code-orange?logo=mathworks&logoColor=white" />
  <img src="https://img.shields.io/badge/Research-Backed-lightgrey" />
  <img src="https://img.shields.io/badge/Academic%20Code-✓-blue" />
  <img src="https://img.shields.io/badge/Reproducibility-✓-brightgreen" />
  </a>
</p>
</div>


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

## ❗ Why It Matters
Real-time control of agile aerial robots is still dominated by **slow NMPC** or **black-box learning-based controllers**. One is too **computationally heavy**, the other is **unsafe without guarantees**.

**KQ-LMPC bridges this gap** by enabling **convex MPC for nonlinear quadrotor dynamics** using Koopman operator theory. This means:
✅ **Real-time feasibility (<10 ms solve time)**  
✅ **Explainable, physics-grounded control**  
✅ **Robustness guarantees (I-ISpS)**  
✅ **Ready for PX4/ROS2 deployment**


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
pip install kq-lmpc-quadrotor
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
<p align="center">
  <a href="https://colab.research.google.com/github/santoshrajkumar/kq-lmpc-quadrotor/blob/main/examples/Notebooks/quick_demo.ipynb">
    <img src="https://img.shields.io/badge/Launch%20Demo-Colab-blue?logo=googlecolab&style=for-the-badge" height="30">
  </a>
</p>


### ⚠️ **Important Dependency Notice**
* This package relies on **acados** for fast Model Predictive Control (MPC).  
* You must configure **acados + Python interface** before running MPC examples.
* Recommended that you install acados and configure, before installing the package if you are going to use MPC.
* Koopman Lifting and LQR will work without acados installation
 * **Quick Setup Checklist**
 - Install acados ✅
 - Enable Python interface ✅
 - Export `ACADOS_SOURCE_DIR` ✅
 - Set library paths:
   - Linux: `LD_LIBRARY_PATH`
   - macOS: `DYLD_LIBRARY_PATH`

* 📚 Install acados: https://docs.acados.org/installation/index.html  
* 🐍 acados Python Interface: https://docs.acados.org/python_interface/index.html  
* 💻 OS Support: **Linux/macOS** (*Not tested on Windows*)

```bash
from kq_lmpc_quadrotor import kqlmpc_demo
kqlmpc_demo()
```

## ⤴️ Koopman Lifting

<p align="center">
  <a href="https://colab.research.google.com/github/santoshrajkumar/kq-lmpc-quadrotor/blob/main/examples/Notebooks/koopman_lifting.ipynb">
    <img src="https://img.shields.io/badge/Launch%20Demo-Colab-blue?logo=googlecolab&style=for-the-badge" height="30">
  </a>
</p>

<p align="center">
  <a href="https://github.com/santoshrajkumar/kq-lmpc-quadrotor/tree/main/MATLAB">
    <img src="https://img.shields.io/badge/MATLAB%20Demo-Code-orange?style=for-the-badge&logo=mathworks&logoColor=white">
  </a>
</p>

## 📊 Benchmarking: KQ-LMPC vs NMPC

* Python 3.10 • Ubuntu 22.04 • AMD Ryzen 3 PRO CPU  
* Metrics: **Mean solve time** (μₜ), **Worst-case solve time** (tᵥ), and **Tracking RMSE** (𝓔ₛ).  
* Prediction horizon **Tₕ ∈ {0.8, 1.4, 2.0, 2.8} s** across **4 tasks**.
* 4 Benchmark Tasks:
  - Task 1 : Follow a Line and Hover
  - Task 2 : Follow a vertically moving helical path.
  - Task 3 : Follow a lemniscate trajectory
  - Task 4 : Follow a knot trajectory.


<center>
<figure style="display:inline-block;margin:0">
  <img src="media/comptime.svg" alt="Your alt text"
       style="max-width:50%;border:1px solid #d0d7de;border-radius:6px;">
  <figcaption style="font-size:0.9em;color:#666;text-align:center;margin-top:6px">
    Computation time comparison (ms)
  </figcaption>
</figure>
</center>

<br>

<center>
<figure style="display:inline-block;margin:0">
  <img src="media/rmse.svg" alt="Your alt text"
       style="max-width:50%;border:1px solid #d0d7de;border-radius:6px;">
  <figcaption style="font-size:0.9em;color:#666;text-align:center;margin-top:6px">
    Tracking RMSE (m) comparison
  </figcaption>
</figure>
</center>

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

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Real-time%20Koopman%20MPC%20for%20quadrotors!%20%F0%9F%9A%81%20Convex%20QP%20MPC%20on%20SE(3)%2C%20no%20neural%20networks.%20Check%20this%20open%20source%20project%20by%20%40SantoshRajkumar%20&url=https://github.com/santoshrajkumar/kq-lmpc-quadrotor">
    <img src="https://img.shields.io/badge/Share_on_Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/santoshrajkumar/kq-lmpc-quadrotor">
    <img src="https://img.shields.io/badge/Share_on_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
</p>
