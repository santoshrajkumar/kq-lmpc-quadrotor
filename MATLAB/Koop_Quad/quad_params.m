%% QuadRotor_paramsUIUC

Ixx = 0.002354405654; % kg-m^2
Iyy = 0.002629495468;
Izz = 0.003186782158;
x_rot = 0.175/2;
y_rot = 0.131/2;

params.m=0.904; % kg
params.J = diag([Ixx Iyy Izz]);
params.d = sqrt(x_rot^2+y_rot^2);
params.k_eta = 0.0000007949116377;
params.c_tau = 1.969658853113417e-8;
params.wr_max = 2750;
params.g=9.81; % g
params.e3=[0;0;1];

