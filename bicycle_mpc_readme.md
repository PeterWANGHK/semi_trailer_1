**High-Level Flow**



main.py drives a simulate–optimize loop: it builds a centre-line reference, calls the MPC to get the first control, rolls the single-track (bicycle) model forward, and logs the result each step (main.py:9-39, main.py:291-319).

mpc.py defines the nonlinear MPC problem on top of CasADi/Ipopt, embedding vehicle dynamics, bounds, and smooth penalties for lane boundaries and obstacles (mpc.py:10-142).

config.py centralizes plant/road limits, weighting terms, and solver settings so the same values configure both the optimizer and visualisation (config.py:6-60).



**config.py**



Horizon length, sample time, and obstacle slots (N=15, dt=0.2, max\_obs=5) set the prediction grid that every other module uses (config.py:6-8).

Vehicle geometry (vehicle\_length, vehicle\_width, L) plus safety distance scale the inflated obstacle ellipses and collision cost (config.py:11-14, config.py:52-54).

Kinematic/actuation limits and steering-rate bounds translate into MPC inequality constraints (config.py:17-24).

Road box (y\_min=-2.5, y\_max=2.5) constrains lateral motion, while start/goal states target a straight lane (config.py:27-31).

Weighting terms balance tracking, smooth control effort, boundary adherence, and terminal accuracy, with large penalties for leaving the lane or approaching obstacles (config.py:38-50).

Simulation length, reference cruise speed, and convergence tolerance drive the closed-loop loop length and Ipopt configuration (config.py:56-60).



**mpc.py**



States collect \[x, y, heading, speed, steering] and controls collect \[acceleration, steering rate], matching the discrete bicycle model integrated with forward-Euler (mpc.py:17-30).

The running cost penalizes deviation from the reference centre-line, large controls, and proximity to road edges through smooth hinge penalties (mpc.py:45-67).

Obstacle avoidance is handled by rotating vehicle–obstacle offsets into each obstacle frame, normalising by combined half-width/length plus safety buffer, and applying an exponential barrier gated by the obstacle-activity flag (mpc.py:69-92).

Terminal cost enforces convergence to the goal pose and velocity (mpc.py:94-99).

Dynamics constraints stitch successive predicted states, with box limits on speed/steering plus actuation bounds and a rate limit on heading change (mpc.py:101-130).

solve injects the latest state, reference trajectory, and predicted obstacle motions (assuming constant-velocity obstacles) into the optimizer and returns optimal controls and the predicted state lattice; failure falls back to a conservative braking action (mpc.py:149-186).



**main.py**



generate\_reference\_path builds a straight-ahead, zero-lateral reference with a speed profile that accelerates, cruises, and then tapers near the goal; steering reference stays zero (main.py:9-39).

Obstacles encode moving rectangles with \[x, y, heading, speed, length, width, active\_flag]; one runs ahead in-lane, the other approaches from the opposite direction to force both an overtake and lane return (main.py:278-281).

Each simulation step recomputes the straight reference, calls mpc.solve, applies only the first control sample through mpc.dynamics, and stops early once the goal tolerance is met (main.py:291-317).

Animation replots the driven path, vehicle footprint, and obstacle motion, occasionally querying the MPC mid-animation for a fresh prediction lattice to visualise how the controller is adapting (main.py:42-205).

Lane Change \& Overtake Mechanism



Because the reference keeps y\_ref=0, any lateral motion is induced solely by the obstacle penalty and road-boundary costs (main.py:13-39, mpc.py:45-92). When the ego vehicle approaches the slower obstacle sitting near y=0.75, the exponential penalty outweighs the lateral tracking cost, pushing the optimizer to steer left within the permitted y\_min/y\_max bounds (config.py:27-29, mpc.py:63-92).

The terminal cost and boundary penalties then pull the trajectory back toward the centreline after clearing the obstacle, while the second obstacle (coming from y=-1.3 with opposite heading) triggers an additional evasive detour before the vehicle recentres to reach the goal (main.py:278-281, mpc.py:94-99).

Constant-velocity obstacle propagation inside mpc.solve ensures the optimizer reasons about where each obstacle will sit across the horizon, enabling predictive lane changes rather than reactive avoidance (mpc.py:155-174).

