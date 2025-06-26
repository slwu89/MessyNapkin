# Scheduling

The `bcsp.jl` problem comes from this Discourse question I answered <https://discourse.julialang.org/t/beginner-basic-cyclic-scheduling-new-to-julia-and-scheduling/129344/1> and `cjsp.jl` from <https://discourse.julialang.org/t/cyclic-job-shop-scheduling/129409>

`BasicSchedule.jl` is largely from:

  * Ulusoy, Gündüz, et al. Introduction to Project Modeling and Planning. Springer International Publishing, 2021.
  * Eiselt, Horst A., and Carl-Louis Sandblom. Decision analysis, location models, and scheduling problems. Springer Science & Business Media, 2013.
  * Eiselt, Horst A., and Carl-Louis Sandblom. Operations research: A model-based approach. Springer Nature, 2022.

`PetriReachability.jl` is from:

  - [Bourdeaud’huy, Thomas, Saïd Hanafi, and Pascal Yim. "Mathematical programming approach to the Petri nets reachability problem." European Journal of Operational Research 177.1 (2007): 176-197.](https://doi.org/10.1016/j.ejor.2005.10.060)
  - [Bourdeaud'Huy, Thomas, et al. "Transient inter-production scheduling based on Petri nets and constraint programming." International journal of production research 49.22 (2011): 6591-6608.](https://doi.org/10.1080/00207543.2010.519113)
  - [Symbolic Scheduling of Robotic Cellular Manufacturing Systems With Timed Petri Nets](https://ieeexplore.ieee.org/abstract/document/9669185?casa_token=r58vEgDaF5gAAAAA:ah1ErePAhwXvZaw15cu9vHBypD_JQj8cv30f9TYdqrDMJ1KgNrvb3krWbaftTfUDP51mP4oOEqs)

  ## Other interesting stuff


  * Basic algorithms in https://github.com/JuliaGraphs/Graphs.jl that arent in Catlab's graph algorithms list
  * Flow problems https://github.com/JuliaGraphs/GraphsFlows.jl
  * check out https://github.com/oxinabox/ProjectManagement.jl
  * check out https://github.com/bprzybylski/Scheduling.jl
