import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle

# Import the configuration and MPC classes
# Make sure these files exist or update the import names accordingly
try:
    from semi_trailer_config import SemiTrailerConfig
except ImportError:
    print("Error: Cannot import SemiTrailerConfig from config.py")
    print("Make sure you have saved the config.py file with SemiTrailerConfig class")
    exit()

try:
    from semi_trailer_mpc import SemiTrailerMPC
except ImportError:
    try:
        from semi_trailer_mpc import SemiTrailerMPC
    except ImportError:
        print("Error: Cannot import SemiTrailerMPC")
        print("Make sure you have saved the mpc.py file with SemiTrailerMPC class")
        exit()


def generate_semi_trailer_reference_path(current_state, goal_state, N, dt, config, obstacles=None):
    """Generate reference trajectory for semi-trailer with realistic highway overtaking"""
    current_x = current_state[0]
    current_y = current_state[1]
    current_v = current_state[3]
    goal_x = goal_state[0]
    goal_y = goal_state[1]  # Should be 0.0 for center lane
    
    x_ref = np.linspace(current_x, goal_x, N + 1)
    y_ref = np.zeros(N + 1)
    theta_t_ref = np.zeros(N + 1)
    theta_s_ref = np.zeros(N + 1)
    
    # Define overtaking phases for REALISTIC HIGHWAY SCENARIO
    FOLLOW = 0        # Following in center lane
    WAIT = 1          # Waiting behind slow vehicle (oncoming traffic in overtaking lane)
    OVERTAKE = 2      # Overtaking in left lane (same lane as oncoming - RISKY!)
    RETURN = 3        # Returning to center lane
    GOAL_APPROACH = 4 # Final approach to goal
    
    # Analyze obstacles for HIGHWAY OVERTAKING scenario
    obstacles_same_direction = []  # Vehicles going same direction (need overtaking)
    obstacles_oncoming = []        # Vehicles coming from opposite direction in OVERTAKING LANE
    
    if obstacles is not None:
        for obs in obstacles:
            if obs[6] < 0.5:  # Skip inactive obstacles
                continue
            
            obs_x0, obs_y0 = obs[0], obs[1]
            obs_v, obs_heading = obs[3], obs[2]
            obs_length = obs[4]
            
            # Determine direction based on heading angle
            if abs(obs_heading) < np.pi/2:  # Heading ~0°, same direction as ego
                # Same direction vehicle (potential overtaking target)
                if (abs(obs_y0) < 1.5 and  # In center lane area
                    obs_x0 > current_x - 5.0):  # Not far behind
                    obstacles_same_direction.append({
                        'x0': obs_x0, 'y0': obs_y0, 'v': obs_v, 'heading': obs_heading,
                        'length': obs_length, 'x_current': obs_x0
                    })
            else:  # Heading ~180°, opposite direction
                # Oncoming vehicle in LEFT LANE (our overtaking lane!)
                if abs(obs_y0 + 2.0) < 1.0:  # Oncoming vehicle in left lane area
                    obstacles_oncoming.append({
                        'x0': obs_x0, 'y0': obs_y0, 'v': obs_v, 'heading': obs_heading,
                        'length': obs_length, 'x_current': obs_x0
                    })
    
    # Generate reference trajectory with TIMING-BASED OVERTAKING
    for i, x_pos in enumerate(x_ref):
        t_future = i * dt
        
        # Determine phase based on position and obstacles
        distance_to_goal = goal_x - x_pos
        
        # Phase determination
        if distance_to_goal < 8.0:
            # GOAL_APPROACH: Force center lane in final approach
            phase = GOAL_APPROACH
            y_target = goal_y  # Must be 0.0 (center lane)
            
        elif len(obstacles_same_direction) == 0:
            # No same-direction obstacles: stay in center or return
            if current_y < -1.0:  # Currently in left lane
                phase = RETURN
                return_distance = 15.0
                return_progress = min(1.0, (x_pos - current_x) / return_distance)
                y_target = -2.2 * (1.0 - return_progress)
            else:
                phase = FOLLOW
                y_target = 0.0
            
        else:
            # Check if we need to overtake same-direction obstacles
            need_overtake = False
            
            for obs in obstacles_same_direction:
                # Predict obstacle position at this time
                obs_x_future = obs['x0'] + obs['v'] * np.cos(obs['heading']) * t_future
                
                # Distance from our future position to obstacle
                distance_to_obs = obs_x_future - x_pos
                
                # Check if obstacle is ahead and blocking
                ego_safety_margin = config.total_length / 2 + 3.0
                obs_safety_margin = obs['length'] / 2 + 3.0
                
                if (distance_to_obs > -ego_safety_margin and 
                    distance_to_obs < obs_safety_margin + 25.0):
                    need_overtake = True
                    break
            
            if need_overtake:
                # CRITICAL: Check if overtaking is SAFE from oncoming traffic
                # This is the FACE-TO-FACE CONFLICT ANALYSIS
                overtaking_safe = True
                time_until_safe = 0.0
                
                for oncoming_obs in obstacles_oncoming:
                    # Predict oncoming vehicle position at overtaking time
                    oncoming_x_future = oncoming_obs['x0'] + oncoming_obs['v'] * np.cos(oncoming_obs['heading']) * t_future
                    
                    # Calculate if we'll have a HEAD-ON COLLISION in left lane
                    ego_front = x_pos + config.total_length / 2
                    ego_rear = x_pos - config.total_length / 2
                    oncoming_front = oncoming_x_future + oncoming_obs['length'] / 2
                    oncoming_rear = oncoming_x_future - oncoming_obs['length'] / 2
                    
                    # CRITICAL SAFETY MARGIN for head-on collision
                    safety_gap = 40.0  # Need 40m separation for semi-trailer overtaking
                    
                    # Check for potential HEAD-ON COLLISION
                    if (ego_rear < oncoming_front + safety_gap and 
                        ego_front > oncoming_rear - safety_gap):
                        overtaking_safe = False
                        
                        # Calculate when it will be safe (oncoming vehicle passes)
                        relative_speed = current_v - oncoming_obs['v'] * np.cos(oncoming_obs['heading'])
                        if relative_speed > 0:
                            time_until_safe = max(0, (oncoming_front - ego_rear + safety_gap) / relative_speed)
                        break
                
                # Decide phase based on FACE-TO-FACE safety analysis
                if overtaking_safe:
                    # SAFE to overtake in left lane
                    phase = OVERTAKE
                    y_target = -2.2  # LEFT lane (same as oncoming direction!)
                else:
                    # DANGEROUS - oncoming vehicle too close
                    phase = WAIT
                    y_target = 0.0   # Stay in center lane behind slow vehicle
            
            elif current_y < -1.0:  # Currently in left lane but obstacles passed
                phase = RETURN
                # Gradual return to center over 15 meters
                return_distance = 15.0
                return_progress = min(1.0, (x_pos - current_x) / return_distance)
                y_target = -2.2 * (1.0 - return_progress)  # Return from left lane
            else:
                phase = FOLLOW
                y_target = 0.0  # Center lane
        
        y_ref[i] = y_target
    
    # Speed reference with OVERTAKING TIMING
    v_ref = np.zeros(N + 1)
    total_distance = max(1.0, goal_x - current_x)
    
    for k in range(N + 1):
        distance_to_goal = goal_x - x_ref[k]
        
        if distance_to_goal < 0.1:
            # AT GOAL: Stop completely
            v_ref[k] = 0.0
        elif distance_to_goal < 3.0:
            # FINAL APPROACH: Slow down to stop
            v_ref[k] = max(2.0, distance_to_goal * 2.0)
        elif distance_to_goal < 8.0:
            # GOAL APPROACH: Moderate speed
            v_ref[k] = min(8.0, config.ref_speed * 0.6)
        else:
            # Check if we're in WAIT phase (need to slow down behind slow vehicle)
            current_phase_here = FOLLOW  # Default
            
            # Quick phase check for this position
            if len(obstacles_same_direction) > 0 and len(obstacles_oncoming) > 0:
                for obs in obstacles_same_direction:
                    distance_to_obs = obs['x0'] - x_ref[k]
                    if distance_to_obs > 0 and distance_to_obs < 30.0:
                        # Check if oncoming prevents overtaking RIGHT NOW
                        for oncoming_obs in obstacles_oncoming:
                            oncoming_distance = abs(oncoming_obs['x0'] - x_ref[k])
                            if oncoming_distance < 50.0:  # Oncoming nearby
                                current_phase_here = WAIT
                                break
            
            # Set speed based on phase
            if current_phase_here == WAIT:
                # WAIT: Slow down to match slow vehicle (don't tailgate)
                slow_vehicle_speed = 4.0  # From obstacle definition
                v_ref[k] = max(slow_vehicle_speed * 0.9, 6.0)  # Slightly slower than obstacle
            elif distance_to_goal > 0.8 * total_distance:
                # ACCELERATION PHASE
                v_ref[k] = min(config.ref_speed, current_v + config.a_max * k * dt)
            elif distance_to_goal > 0.3 * total_distance:
                # CRUISING PHASE
                v_ref[k] = config.ref_speed
            else:
                # DECELERATION PHASE
                decel_factor = distance_to_goal / (0.3 * total_distance)
                v_ref[k] = config.ref_speed * max(0.4, decel_factor)
    
    delta_ref = np.zeros(N + 1)
    
    return np.vstack([x_ref, y_ref, theta_t_ref, v_ref, delta_ref, theta_s_ref])


def animate_semi_trailer_trajectory(state_history, obstacles, mpc, config, n_steps):
    """Animation function updated for semi-trailer"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.set_aspect('equal')
    ax.set_title("Semi-Trailer MPC Obstacle Avoidance", fontsize=16)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    # Draw road boundaries and center line
    ax.axhline(config.y_min, color='gray', linestyle='-', linewidth=3, label='Road Boundary')
    ax.axhline(config.y_max, color='gray', linestyle='-', linewidth=3)
    ax.axhline(0, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Road Centerline')

    # Start and goal markers
    ax.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=15, label='Goal')
    
    # Trajectory line
    path_line, = ax.plot([], [], 'b-', lw=2, alpha=0.7, label='Actual Path')
    
    # Semi-trailer visualization
    tractor_patch = Rectangle((0, 0), config.tractor_length, config.vehicle_width, 
                             angle=0, color='blue', alpha=0.8, label='Tractor')
    trailer_patch = Rectangle((0, 0), config.trailer_length, config.vehicle_width, 
                             angle=0, color='lightblue', alpha=0.7, label='Trailer')
    ax.add_patch(tractor_patch)
    ax.add_patch(trailer_patch)
    
    # Articulation line (connection between tractor and trailer)
    articulation_line, = ax.plot([], [], 'r-', lw=3, alpha=0.8, label='Articulation')
    
    # Obstacle visualization
    obstacle_patches = []
    safety_boundaries = []
    for obs in obstacles:
        if obs[6] < 0.5:
            continue  # Skip inactive obstacles
        
        x0, y0, heading, v_obs, length, width, _ = obs
        # Obstacle rectangle
        obs_patch = Rectangle((0, 0), length, width, angle=np.rad2deg(heading), 
                             color='red', alpha=0.6)
        ax.add_patch(obs_patch)
        obstacle_patches.append(obs_patch)
        
        # Safety boundary ellipse
        ellipse_width = length + config.total_length + 2*config.safety_distance
        ellipse_height = width + config.vehicle_width + 2*config.safety_distance
        ellipse = Ellipse((x0, y0), width=ellipse_width, height=ellipse_height,
                         angle=np.rad2deg(heading), fill=False, edgecolor='purple',
                         linestyle='--', linewidth=1, alpha=0.3, label='Safety Boundary')
        ax.add_patch(ellipse)
        safety_boundaries.append(ellipse)

    # Predicted trajectory
    pred_line, = ax.plot([], [], 'm--', lw=2, alpha=0.6, label='Predicted Path')
    pred_points = ax.plot([], [], 'mo', markersize=4, alpha=0.6)[0]
    
    # Reference trajectory
    ref_line, = ax.plot([], [], 'g:', lw=2, alpha=0.8, label='Reference Path')

    # Information text
    info_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def get_semi_trailer_positions(x, y, theta_t, theta_s):
        """Calculate tractor and trailer positions and orientations"""
        # Tractor rear axle is at (x, y)
        # Tractor front is forward from rear axle
        tractor_front_x = x + config.L_t * np.cos(theta_t)
        tractor_front_y = y + config.L_t * np.sin(theta_t)
        
        # Tractor center for visualization
        tractor_center_x = x + config.L_t/2 * np.cos(theta_t)
        tractor_center_y = y + config.L_t/2 * np.sin(theta_t)
        
        # Trailer is behind tractor rear axle
        trailer_front_x = x
        trailer_front_y = y
        trailer_rear_x = x - config.L_s * np.cos(theta_s)
        trailer_rear_y = y - config.L_s * np.sin(theta_s)
        
        # Trailer center for visualization
        trailer_center_x = x - config.L_s/2 * np.cos(theta_s)
        trailer_center_y = y - config.L_s/2 * np.sin(theta_s)
        
        return (tractor_center_x, tractor_center_y, theta_t,
                trailer_center_x, trailer_center_y, theta_s,
                tractor_front_x, tractor_front_y, trailer_rear_x, trailer_rear_y)

    def init_animation():
        path_line.set_data([], [])
        tractor_patch.set_xy((0, 0))
        trailer_patch.set_xy((0, 0))
        articulation_line.set_data([], [])
        pred_line.set_data([], [])
        pred_points.set_data([], [])
        ref_line.set_data([], [])
        info_text.set_text('')
        return ([path_line, tractor_patch, trailer_patch, articulation_line, 
                pred_line, pred_points, ref_line, info_text] + obstacle_patches + safety_boundaries)

    def update_animation(frame):
        idx = min(frame, len(state_history)-1)
        current_state = state_history[idx]
        x, y, theta_t, v, delta, theta_s = current_state
        
        # Update trajectory
        path_line.set_data(state_history[:idx+1, 0], state_history[:idx+1, 1])
        
        # Get vehicle positions
        (tractor_cx, tractor_cy, theta_t,
         trailer_cx, trailer_cy, theta_s,
         tractor_fx, tractor_fy, trailer_rx, trailer_ry) = get_semi_trailer_positions(x, y, theta_t, theta_s)
        
        # Update tractor
        tractor_patch.set_xy((tractor_cx - config.tractor_length/2, 
                             tractor_cy - config.vehicle_width/2))
        tractor_patch.set_angle(np.rad2deg(theta_t))
        
        # Update trailer
        trailer_patch.set_xy((trailer_cx - config.trailer_length/2, 
                             trailer_cy - config.vehicle_width/2))
        trailer_patch.set_angle(np.rad2deg(theta_s))
        
        # Update articulation line (connection point)
        articulation_line.set_data([tractor_cx - config.tractor_length/2 * np.cos(theta_t),
                                   trailer_cx + config.trailer_length/2 * np.cos(theta_s)],
                                  [tractor_cy - config.tractor_length/2 * np.sin(theta_t),
                                   trailer_cy + config.trailer_length/2 * np.sin(theta_s)])
        
        # Calculate articulation angle
        phi = theta_t - theta_s
        
        # Update information text
        info_text.set_text(
            f"Time: {idx*config.dt:.1f}s\n"
            f"Speed: {v:.1f}m/s ({v*3.6:.0f}km/h)\n"
            f"Tractor Heading: {np.rad2deg(theta_t):.1f}°\n"
            f"Trailer Heading: {np.rad2deg(theta_s):.1f}°\n"
            f"Articulation: {np.rad2deg(phi):.1f}°\n"
            f"Steering: {np.rad2deg(delta):.1f}°"
        )
        
        # Update obstacle positions (dynamic obstacles)
        for obs_idx, obs in enumerate(obstacles):
            if obs[6] < 0.5:
                continue
            
            x0, y0, heading, v_obs, length, width, _ = obs
            t = config.dt * idx
            # Obstacle current position
            x_obs = x0 + np.cos(heading) * v_obs * t
            y_obs = y0 + np.sin(heading) * v_obs * t
            
            # Update obstacle rectangle
            obstacle_patches[obs_idx].set_xy((x_obs - length/2, y_obs - width/2))
            
            # Update safety boundary
            safety_boundaries[obs_idx].center = (x_obs, y_obs)
        
        # Update predicted trajectory every few frames
        if frame % 3 == 0 and idx < len(state_history)-1:
            try:
                ref_traj = generate_semi_trailer_reference_path(
                    current_state, config.goal_pos, mpc.config.N, config.dt, config, obstacles)
                
                # Show reference trajectory
                ref_line.set_data(ref_traj[0, :], ref_traj[1, :])
                
                _, X_pred = mpc.solve(current_state, ref_traj, obstacles)
                pred_line.set_data(X_pred[0, :], X_pred[1, :])
                pred_points.set_data(X_pred[0, :], X_pred[1, :])
            except:
                pass  # Skip prediction update if solve fails
        
        return ([path_line, tractor_patch, trailer_patch, articulation_line, 
                pred_line, pred_points, ref_line, info_text] + obstacle_patches + safety_boundaries)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update_animation, frames=n_steps+1, init_func=init_animation,
        interval=config.animation_interval, blit=True, repeat=False
    )
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
             bbox_to_anchor=(1.0, 1.0), ncol=2)
    
    # Save animation
    try:
        ani.save('semi_trailer_mpc_animation.gif', writer='pillow', fps=10)
        print("Animation saved as 'semi_trailer_mpc_animation.gif'")
    except:
        print("Could not save animation (pillow/imagemagick not available)")
    
    plt.tight_layout()
    plt.show()
    return ani


def plot_semi_trailer_results(state_history, pred_history, config):
    """Plot analysis results for semi-trailer"""
    plt.figure(figsize=(15, 12))
    
    # 1. Trajectory plot
    plt.subplot(3, 2, 1)
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', linewidth=2, label='Actual Path')
    plt.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='Goal')
    
    # Plot some predicted trajectories
    for i, pred in enumerate(pred_history):
        if i % 10 == 0:
            plt.plot(pred[0, :], pred[1, :], 'm--', alpha=0.3)
    
    plt.axhline(config.y_min, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(config.y_max, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Semi-Trailer Trajectory')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(config.xlim)
    plt.ylim(config.ylim)
    
    # 2. Speed and heading evolution
    plt.subplot(3, 2, 2)
    time = np.arange(len(state_history)) * config.dt
    plt.plot(time, state_history[:, 3] * 3.6, 'b-', label='Speed (km/h)')
    plt.plot(time, np.rad2deg(state_history[:, 2]), 'g-', label='Tractor Heading (°)')
    plt.plot(time, np.rad2deg(state_history[:, 5]), 'c-', label='Trailer Heading (°)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Speed and Heading Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Articulation angle (most important for semi-trailer)
    plt.subplot(3, 2, 3)
    phi = np.rad2deg(state_history[:, 2] - state_history[:, 5])
    plt.plot(time, phi, 'r-', linewidth=2, label='Articulation Angle (°)')
    plt.axhline(60, color='r', linestyle='--', alpha=0.5, label='Limit (±60°)')
    plt.axhline(-60, color='r', linestyle='--', alpha=0.5)
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Articulation Angle (°)')
    plt.title('Articulation Angle (Jackknife Prevention)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Control inputs
    plt.subplot(3, 2, 4)
    if len(state_history) > 1:
        acceleration = np.diff(state_history[:, 3]) / config.dt
        delta_dot = np.diff(np.rad2deg(state_history[:, 4])) / config.dt
        time_ctrl = np.arange(len(acceleration)) * config.dt
        
        plt.plot(time_ctrl, acceleration, 'r-', label='Acceleration (m/s²)')
        plt.plot(time_ctrl, delta_dot, 'c-', label='Steering Rate (°/s)')
        plt.axhline(config.a_max, color='r', linestyle='--', alpha=0.5)
        plt.axhline(config.a_min, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Signals')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 5. Lateral error from centerline
    plt.subplot(3, 2, 5)
    lateral_error = state_history[:, 1]
    plt.plot(time, lateral_error, 'm-', linewidth=2, label='Lateral Error (m)')
    plt.axhline(0, color='g', linestyle='--', alpha=0.5, label='Centerline')
    plt.axhline(config.y_max, color='r', linestyle='--', alpha=0.5, label='Road Boundary')
    plt.axhline(config.y_min, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Lateral Position (Lane Keeping)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Vehicle path curvature
    plt.subplot(3, 2, 6)
    if len(state_history) > 2:
        # Calculate path curvature
        dx = np.diff(state_history[:, 0])
        dy = np.diff(state_history[:, 1])
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        curvature = np.abs(dx[1:] * ddy - dy[1:] * ddx) / (dx[1:]**2 + dy[1:]**2)**(3/2)
        curvature = np.nan_to_num(curvature)  # Handle division by zero
        
        time_curv = np.arange(len(curvature)) * config.dt
        plt.plot(time_curv, curvature, 'orange', linewidth=2, label='Path Curvature (1/m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Curvature (1/m)')
        plt.title('Path Curvature')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def create_semi_trailer_scenario():
    """Create semi-trailer overtaking scenario"""
    # Obstacles: [x0, y0, heading, speed, length, width, active_flag]
    obstacles = [
        # Slow truck ahead (in same lane) - VERY slow for easy overtaking
        [18.0, 1.2, np.deg2rad(0), 1.0, 4.5, 2.5, 1.0],     # slow front vehicle
        
        # Oncoming car (in opposite lane) - much further away
        [40.0, -2.2, np.deg2rad(180), 20.0, 4.5, 2.0, 1.0]   # fast oncoming car (further away)
    ]
    
    return obstacles


def main():
    print("Semi-Trailer MPC Obstacle Avoidance Simulation")
    print("=" * 50)
    
    # Initialize configuration and MPC controller
    config = SemiTrailerConfig()
    mpc = SemiTrailerMPC(config)
    
    # Initial state: [x, y, θ_tractor, v, δ, θ_trailer]
    initial_state = np.array([
        config.start_pos[0],  # x position
        config.start_pos[1],  # y position  
        config.start_pos[2],  # tractor heading angle
        config.start_pos[3],  # speed
        0.0,                  # steering angle (initial)
        config.start_pos[5]   # trailer heading angle (aligned with tractor)
    ])
    
    # Create semi-trailer scenario
    obstacles = create_semi_trailer_scenario()
    
    print(f"Vehicle Configuration:")
    print(f"  - Total Length: {config.total_length:.1f}m")
    print(f"  - Tractor: {config.tractor_length:.1f}m, Trailer: {config.trailer_length:.1f}m")
    print(f"  - Max Speed: {config.v_max*3.6:.0f} km/h")
    print(f"  - Max Acceleration: {config.a_max:.1f} m/s²")
    print(f"  - Max Articulation: ±{np.rad2deg(config.phi_max):.0f}°")
    print()
    
    print(f"Speed Configuration:")
    print(f"  - Ego Semi-Trailer: {initial_state[3]*3.6:.0f} km/h ({initial_state[3]:.1f} m/s)")
    
    for i, obs in enumerate(obstacles):
        if obs[6] > 0.5:  # Active obstacle
            if i == 0:
                print(f"  - Slow Vehicle:     {obs[3]*3.6:.0f} km/h ({obs[3]:.1f} m/s) - DIFFERENCE: {(initial_state[3]-obs[3])*3.6:.0f} km/h")
            else:
                print(f"  - Oncoming Vehicle: {obs[3]*3.6:.0f} km/h ({obs[3]:.1f} m/s)")
    print()
    
    print(f"Scenario:")
    print(f"  - Road Length: {config.road_length:.0f}m")
    print(f"  - Lane Width: {config.y_max - config.y_min:.1f}m")
    print(f"  - Number of Obstacles: {len([obs for obs in obstacles if obs[6] > 0.5])}")
    print()
    
    # Simulation initialization
    n_steps = int(config.sim_time / config.dt)
    state_history = np.zeros((n_steps + 1, 6))  # 6 states for semi-trailer
    state_history[0] = initial_state
    current_state = initial_state.copy()
    pred_history = []

    print("Starting simulation...")
    print(f"Total steps: {n_steps}, Time step: {config.dt}s")
    print()

    successful_steps = n_steps
    for i in range(n_steps):
        # Generate reference trajectory with obstacle awareness
        ref_traj = generate_semi_trailer_reference_path(
            current_state, config.goal_pos, config.N, config.dt, config, obstacles
        )
        
        # Solve MPC
        try:
            u_opt, x_pred = mpc.solve(current_state, ref_traj, obstacles)
            pred_history.append(x_pred)
            
            # Apply first control action
            control = u_opt[:, 0]
            next_state = mpc.dynamics(current_state, control).full().flatten()
            
            current_state = next_state
            state_history[i+1] = current_state
            
            # Calculate articulation angle
            phi = current_state[2] - current_state[5]
            
            # Progress report
            if i % 20 == 0 or i < 10:
                print(f"Step {i+1:3d}/{n_steps}: "
                      f"Pos=({current_state[0]:5.1f}, {current_state[1]:5.1f})m, "
                      f"Speed={current_state[3]*3.6:4.0f}km/h, "
                      f"φ={np.rad2deg(phi):5.1f}°")
            
            # Check goal reached
            pos_error = np.linalg.norm(current_state[:2] - config.goal_pos[:2])
            if pos_error < config.goal_threshold:
                print(f"\n🎯 Goal reached! Position error: {pos_error:.2f}m")
                print(f"   Completion time: {(i+1)*config.dt:.1f}s")
                successful_steps = i + 1
                break
                
            # Check articulation safety
            if abs(phi) > config.phi_max * 0.9:
                print(f"\n⚠️  High articulation angle: {np.rad2deg(phi):.1f}°")
            
        except Exception as e:
            print(f"\n❌ MPC solve failed at step {i+1}: {e}")
            print("   Stopping simulation...")
            successful_steps = i
            break

    print(f"\nSimulation completed!")
    print(f"Final position: ({current_state[0]:.1f}, {current_state[1]:.1f})m")
    print(f"Final speed: {current_state[3]*3.6:.0f} km/h")
    print(f"Total time: {successful_steps*config.dt:.1f}s")
    
    # Trim history to successful steps
    state_history = state_history[:successful_steps+1]
    
    # Calculate performance metrics
    max_articulation = np.max(np.abs(np.rad2deg(state_history[:, 2] - state_history[:, 5])))
    max_lateral_dev = np.max(np.abs(state_history[:, 1]))
    avg_speed = np.mean(state_history[:, 3]) * 3.6
    
    print(f"\nPerformance Metrics:")
    print(f"  - Max Articulation Angle: {max_articulation:.1f}° (limit: ±{np.rad2deg(config.phi_max):.0f}°)")
    print(f"  - Max Lateral Deviation: {max_lateral_dev:.2f}m")
    print(f"  - Average Speed: {avg_speed:.0f} km/h")
    print(f"  - Distance Traveled: {current_state[0] - initial_state[0]:.1f}m")

    # Create visualizations
    print("\nGenerating animation...")
    animate_semi_trailer_trajectory(state_history, obstacles, mpc, config, successful_steps)
    
    print("Generating analysis plots...")
    plot_semi_trailer_results(state_history, pred_history, config)
    
    print("\n✅ Semi-trailer MPC simulation completed successfully!")


if __name__ == "__main__":
    main()