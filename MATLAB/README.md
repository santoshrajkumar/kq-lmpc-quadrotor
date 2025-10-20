## Koopman Lifting in MATLAB

- **We provided Koopman Lifting Functionalities Only**
- MATLAB code is **not actively worked on**, and hence **no longer supported**.
- The authors have **acados LMPC implementation in MATLAB, available upon request**.
- **Contributions** on MATLAB side are encouraged.

---

### 🔧 Construct a dummy 18D state vector

```matlab
% Construct a dummy 18D state vector
R = [0.0996   -0.1422    0.9848;
     0.8988   -0.4118   -0.1504;
     0.4270    0.9001    0.0868];
x=[randn(6,1); R(:); randn(3,1)];
disp(x)
```
Output:

```bash
    1.7118
   -0.1943
   -2.1384
   -0.8396
    1.3546
   -1.0721
    0.0996
    0.8988
    0.4270
   -0.1422
   -0.4118
    0.9001
    0.9848
   -0.1504
    0.0868
    0.9610
    0.1240
    1.4367

```

### Imports and Initialization

```matlab
addpath Koop_Quad/

import KoopmanLift.*
quad_params % loads quadrotor parameters as params
M=3; N=2;
koop=KoopmanLift(params,M,N); % initialize the class

```

### Koopman Lifting

```matlab
% construct the lifted state
X=koop.fcn_gen_koopman_states_se3(x);
disp(size(X))
```
Output

```bash
    45     1
```

```matlab
% get back the actual state
x_rec = koop.fcn_se3_states_to_actual(X);
disp(x_rec)
```
```bash
    1.7118
   -0.1943
   -2.1384
   -0.8396
    1.3546
   -1.0721
    0.0996
    0.8988
    0.4270
   -0.1422
   -0.4118
    0.9001
    0.9848
   -0.1504
    0.0868
    0.9610
    0.1240
    1.4367

```

### Construct System Matrices LTI/LPV

```matlab
% Construct A matrix
A = koop.fcn_A_lifted();
disp(size(A))

% Construct LTI B matrix
Bbar = koop.fcn_Bbar();
disp(size(Bbar))

% Construct calB (state dependent B)
calB  = koop.fcn_CalB(x,X);
disp(size(calB))

```
output
```bash
    45    45

    45    28

    45     4

```