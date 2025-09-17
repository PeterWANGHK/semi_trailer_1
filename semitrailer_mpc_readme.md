**Config Layer**



SemiTrailerConfig expands the bicycle settings to a 6‑state tractor–trailer rig, supplying distinct tractor/trailer lengths, hitch distances, and a nonzero safety buffer used by both the planner and the renderer (semi\_trailer\_config.py:6-59).

Road geometry and scenario bounds widen to a highway-scale stage, while start/goal vectors now spell \[x, y, θ\_tractor, v, δ, θ\_trailer] so downstream code knows the desired articulation-free pose (semi\_trailer\_config.py:35-41).

Kinematic limits tighten (e.g., reduced steering angle/rate) and new articulation angle bounds (phi\_min/max) govern how far the trailer may swing relative to the tractor; these limits are knitted into MPC constraints (semi\_trailer\_config.py:21-33).

Weighting terms emphasise lateral discipline, boundary compliance for both units, and explicit penalties on articulation, especially at the terminal state to guarantee the trailer straightens out after a maneuver (semi\_trailer\_config.py:48-64).



**MPC Formulation**



SemiTrailerMPC promotes the state vector to \[x, y, θ\_t, v, δ, θ\_s] and retains two controls \[a, delta\_dot], embedding a kinematic single-track tractor and a trailer kinematic update θ̇\_s = v·sin(phi)/L\_s with articulation phi = θ\_t − θ\_s (semi\_trailer\_mpc.py:11-36).

The running cost tracks reference pose/speed, penalises control effort, and adds articulation and dual boundary penalties so both the tractor front and trailer tail stay within the lane box (semi\_trailer\_mpc.py:52-83).

Obstacle avoidance computes separate rotated distances for the tractor front and the trailer axle, inflating footprints by their respective half-lengths and safety margin, then sums the smooth exponential penalties gated by each obstacle’s active flag (semi\_trailer\_mpc.py:85-121).

Terminal terms reinforce convergence in pose/speed plus a heavier articulation penalty to finish straight (semi\_trailer\_mpc.py:123-133).

Constraints enforce dynamics, velocity/steering bounds, articulation limits, actuator limits, and a heading-rate cap, closely mirroring the config limits (semi\_trailer\_mpc.py:140-173).

solve mirrors the bicycle case: populate parameters, roll constant-velocity obstacle predictions across the horizon, and fall back to a braking trajectory if Ipopt fails (semi\_trailer\_mpc.py:192-249).



**Simulation \& Reference Logic**



semi\_trailer\_v2.py’s generate\_semi\_trailer\_reference\_path partitions oncoming traffic into “same-direction” blockers vs. oncoming threats, then assigns phases (FOLLOW, WAIT, OVERTAKE, RETURN, GOAL\_APPROACH) to craft lateral and speed references that mimic a cautious highway overtake (semi\_trailer\_v2.py:26-222).

During OVERTAKE, it commands y≈-2.2 (left lane) only if a 40 m safety margin to oncoming traffic is predicted; otherwise the reference stays in-lane and slows (WAIT phase) until the head-on conflict clears (semi\_trailer\_v2.py:101-209).

Speed references accelerate toward ref\_speed, cruise, or decelerate based on distance-to-goal and the detected phase, ensuring slowing behind a slow lead vehicle and tapering near the finish (semi\_trailer\_v2.py:173-219).

create\_semi\_trailer\_scenario instantiates a slow in-lane truck plus a fast oncoming car to trigger the phased logic (semi\_trailer\_v2.py:539-547).

main wires everything: instantiate config/MPC, seed the 6‑state initial condition, fetch the obstacle list, then iterate the MPC loop—generate ref, solve, apply first control via the CasADi dynamics, log articulation, and exit once the planar position matches the goal tolerance (semi\_trailer\_v2.py:550-678).

Post-loop diagnostics report final pose, articulation metrics, lateral deviation, average speed, and then drive the animation/plot routines which visualise both tractor and trailer footprints, predicted lattices, and obstacle motion.



Together these three modules extend the bicycle framework to a full articulated-vehicle MPC: the config supplies richer geometry/limits, the MPC adds articulation-aware dynamics and penalties, and the driver builds sophisticated overtaking references while feeding the optimizer with moving obstacles so the semi-trailer can safely change lanes, pass, and re-centre.

