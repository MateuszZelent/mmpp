# STNO Vortex Current Sweep Template

This template is the cleaned MuMax3 starting point for extracting the vortex
gyrotropic frequency versus CPP current and overlaying it with the
field-resolved Thiele model in MMPP.

Key conventions:

- `I_pillar_mA` is the sweep variable.
- `Jdc = I_pillar_mA * 1e-3 / Area`.
- Fields are specified as `Bx_mT`, `By_mT`, `Bz_mT` and converted to tesla.
- `addOe=0` and `addFL=0` match the baseline overlay assumptions.
- `ext_corepos` is saved for `job[:].vortex.frequency_sweep(...)`.

For analytical overlays, preserve `D`, `Area`, `FreeLayerThickness`, `Pol`,
`Lambda`, `epsilonprime`, `FixedLayerPosition`, and `FixedLayer` in metadata.
The reduced MuMax3 mapping uses the time/space representative
`mean_m_dot_p = mean(Dot(m, FixedLayer))` for the angular efficiency; it is not
equal to `FixedLayer.z` for a tilted layer. The magnetic thickness used by the
gyrovector and the torque-active thickness must be supplied separately if they
differ.

```go
addOe := 0
addFL := 0

Ni := 324
th_pillar := 9e-9

Nx = Ni/2
Ny = Ni/2
Nz = 1

Tx = Ni * 1e-9
Ty = Ni * 1e-9
Tz = th_pillar

SmoothMesh(true, true, false)
EdgeSmooth = 8

D := Ni * 1e-9
SetGeom(circle(D))

Msat = 800e3
Aex = 10e-12
alpha = 0.013

DisableZhangLiTorque = true
Pol = 0.5
Lambda = 1.0
epsilonprime = 0.1
FixedLayerPosition = FIXEDLAYER_BOTTOM

phi_deg := 0.0
theta_deg := 90.0
phi := phi_deg * pi / 180
theta := theta_deg * pi / 180

px := cos(theta) * cos(phi)
py := cos(theta) * sin(phi)
pz := sin(theta)
n_len := sqrt(px*px + py*py + pz*pz)
FixedLayer = vector(px/n_len, py/n_len, pz/n_len)

Bx_mT := 0.0021
By_mT := 0.0
Bz_mT := 0.120

kFL_mT_per_mA := 0.1
Ifl_dc_mA := 0.0
Ifl_ac_mA := 0.0
f_fl := 200e6
By_fl_dc_T := kFL_mT_per_mA * Ifl_dc_mA * 1e-3

B_ext = vector(Bx_mT*1e-3, By_mT*1e-3 + By_fl_dc_T, Bz_mT*1e-3)

chir := 1
pola := 1
x0_nm := 1.0
y0_nm := 0.0

J = vector(0, 0, 0)
EnableOersted = false
J_oersted = vector(0, 0, 0)

m = vortex(chir, pola).transl(x0_nm*1e-9, y0_nm*1e-9, 0)

I_pillar_mA := 0.0
Area := pi * (D/2) * (D/2)
Jdc := I_pillar_mA * 1e-3 / Area

Jac := 0.0
fcut := 10e9
t0 := 200e-12
J = vector(0, 0, Jdc + Jac*sinc(2*pi*fcut*(t - t0)))

if addOe == 1 {
    print("Enable Oersted field")
    J_oersted = vector(0, 0, Jdc + Jac*sinc(2*pi*fcut*(t - t0)))
    EnableOersted = true
}

if addFL == 1 {
    B_ext = vector(
        Bx_mT*1e-3,
        By_mT*1e-3 + kFL_mT_per_mA*(Ifl_dc_mA + Ifl_ac_mA*sin(2*pi*f_fl*t))*1e-3,
        Bz_mT*1e-3
    )
} else {
    B_ext = vector(Bx_mT*1e-3, By_mT*1e-3 + By_fl_dc_T, Bz_mT*1e-3)
}

TableAdd(B_ext)
TableAdd(B_oersted)
TableAdd(E_total)
TableAdd(E_exch)
TableAdd(E_demag)
TableAdd(ext_corepos)
TableAdd(m)
TableAdd(Dot(m, FixedLayer))

f_cut := 8e9
sample_count := 1000
sampling_interval := 0.5 / (f_cut * 1.6)
run_time := sample_count * sampling_interval

TableAutoSave(sampling_interval)
MaxErr = 1e-5
run(600e-9)
AutoSave(m, sampling_interval)
Run(run_time)
SaveAs(m, "end")
```
