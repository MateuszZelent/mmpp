STNO Vortex Current Sweep Template
==================================

Cleaned MuMax3 starting point for extracting the vortex gyrotropic frequency
versus CPP current and overlaying it with the field-resolved Thiele model.

Conventions
-----------

* ``I_pillar_mA`` is the sweep variable.
* ``Jdc = I_pillar_mA * 1e-3 / Area``.
* ``Bx_mT``, ``By_mT`` and ``Bz_mT`` are converted to tesla.
* ``addOe = 0`` and ``addFL = 0`` match the baseline model overlay.
* ``ext_corepos`` is saved for ``job[:].vortex.frequency_sweep(...)``.

Template
--------

.. code-block:: go

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

   FixedLayer = vector(0, 0, 1)

   Bx_mT := 0.0021
   By_mT := 0.0
   Bz_mT := 0.120
   B_ext = vector(Bx_mT*1e-3, By_mT*1e-3, Bz_mT*1e-3)

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
   J = vector(0, 0, Jdc)

   if addOe == 1 {
       J_oersted = vector(0, 0, Jdc)
       EnableOersted = true
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
