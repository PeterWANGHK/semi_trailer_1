import numpy as np
from casadi import SX, vertcat, cos, sin, tan, Function, Opti, log, exp


class SemiTrailerMPC:
    def __init__(self, config):
        self.config = config
        self.setup_optimizer()

    def setup_optimizer(self):
        # 状态变量：x(位置), y(位置), theta_t(牵引车航向), v(速度), delta(转向角), theta_s(挂车航向)
        x, y, theta_t, v, delta, theta_s = (SX.sym('x'), SX.sym('y'), SX.sym('theta_t'), 
                                            SX.sym('v'), SX.sym('delta'), SX.sym('theta_s'))

        # 控制变量：a(加速度), delta_dot(转向角速率)
        a, delta_dot = SX.sym('a'), SX.sym('delta_dot')
        
        self.states = vertcat(x, y, theta_t, v, delta, theta_s)  # 6个状态
        self.controls = vertcat(a, delta_dot)                    # 2个控制

        # 半挂车运动学模型
        phi = theta_t - theta_s  # 铰接角 (articulation angle)
        
        # 状态更新方程
        states_dot = vertcat(
            v * cos(theta_t),           # 牵引车后轴x方向速度
            v * sin(theta_t),           # 牵引车后轴y方向速度
            v * tan(delta) / self.config.L_t,  # 牵引车偏航率
            a,                          # 速度变化率
            delta_dot,                  # 转向角变化率
            v * sin(phi) / self.config.L_s  # 挂车偏航率 (关键新增项!)
        )
        
        # 离散化动力学模型
        self.dynamics = Function('dynamics', [self.states, self.controls], 
                                 [self.states + states_dot * self.config.dt])

        self.opti = Opti()

        # 预测时域内的状态和控制序列
        self.predicted_states = self.opti.variable(6, self.config.N + 1)  # 6个状态
        self.controls_seq = self.opti.variable(2, self.config.N)

        # 优化问题参数
        self.initial_state = self.opti.parameter(6)                       # 6个状态
        self.reference_trajectory = self.opti.parameter(6, self.config.N + 1)
        self.obstacles = self.opti.parameter(7 * self.config.max_obs, self.config.N + 1)

        self.cost = 0

        # 成本函数构建
        for k in range(self.config.N):
            state = self.predicted_states[:, k]
            ref_state = self.reference_trajectory[:, k]

            # 基本轨迹跟踪成本
            self.cost += self.config.w_x * (state[0] - ref_state[0])**2       # x位置误差
            self.cost += self.config.w_y * (state[1] - ref_state[1])**2       # y位置误差
            self.cost += self.config.w_theta * (state[2] - ref_state[2])**2   # 牵引车航向误差
            self.cost += self.config.w_v * (state[3] - ref_state[3])**2       # 速度误差

            # 铰接角惩罚 (防止折刀)
            phi = state[2] - state[5]  # theta_t - theta_s
            self.cost += self.config.w_articulation * phi**2

            # 控制平滑成本
            control = self.controls_seq[:, k]
            self.cost += self.config.w_a * control[0]**2
            self.cost += self.config.w_delta_dot * control[1]**2

            # 道路边界惩罚 - 牵引车
            tractor_y = state[1]
            boundary_penalty_left = self.smooth_max(self.config.y_min - tractor_y)
            boundary_penalty_right = self.smooth_max(tractor_y - self.config.y_max)
            self.cost += self.config.w_bound * (boundary_penalty_left**2 + boundary_penalty_right**2)
            
            # 道路边界惩罚 - 挂车 (计算挂车后端位置)
            trailer_x = state[0] - self.config.L_s * cos(state[5])
            trailer_y = state[1] - self.config.L_s * sin(state[5])
            trailer_boundary_left = self.smooth_max(self.config.y_min - trailer_y)
            trailer_boundary_right = self.smooth_max(trailer_y - self.config.y_max)
            self.cost += self.config.w_bound * (trailer_boundary_left**2 + trailer_boundary_right**2)

            # 障碍物规避惩罚 - 检查牵引车和挂车两部分
            for obs_idx in range(self.config.max_obs):
                obs_params = self.obstacles[obs_idx*7 : (obs_idx+1)*7, k]
                obs_x, obs_y, obs_heading = obs_params[0], obs_params[1], obs_params[2]
                obs_length, obs_width = obs_params[3], obs_params[4]
                obs_flag = obs_params[6]

                # 牵引车前端碰撞检查
                tractor_front_x = state[0] + self.config.L_t * cos(state[2])
                tractor_front_y = state[1] + self.config.L_t * sin(state[2])
                
                dx_t = tractor_front_x - obs_x
                dy_t = tractor_front_y - obs_y
                # 转换到障碍物坐标系
                rot_dx_t = dx_t * cos(obs_heading) + dy_t * sin(obs_heading)
                rot_dy_t = -dx_t * sin(obs_heading) + dy_t * cos(obs_heading)

                # 牵引车安全边界
                margin_x_t = obs_length/2 + self.config.tractor_length/2 + self.config.safety_distance
                margin_y_t = obs_width/2 + self.config.vehicle_width/2 + self.config.safety_distance
                norm_dist_t = (rot_dx_t / margin_x_t)**2 + (rot_dy_t / margin_y_t)**2

                # 挂车后端碰撞检查
                dx_s = trailer_x - obs_x
                dy_s = trailer_y - obs_y
                rot_dx_s = dx_s * cos(obs_heading) + dy_s * sin(obs_heading)
                rot_dy_s = -dx_s * sin(obs_heading) + dy_s * cos(obs_heading)

                # 挂车安全边界
                margin_x_s = obs_length/2 + self.config.trailer_length/2 + self.config.safety_distance
                margin_y_s = obs_width/2 + self.config.vehicle_width/2 + self.config.safety_distance
                norm_dist_s = (rot_dx_s / margin_x_s)**2 + (rot_dy_s / margin_y_s)**2

                # 分别对牵引车和挂车施加障碍物惩罚
                obstacle_penalty_t = exp(-self.config.obstacle_decay_rate * (norm_dist_t - 1))
                obstacle_penalty_s = exp(-self.config.obstacle_decay_rate * (norm_dist_s - 1))
                total_penalty = obstacle_penalty_t + obstacle_penalty_s
                self.cost += self.config.w_obstacle * self.smooth_max(obs_flag - 0.5) * total_penalty

        # 终端成本
        final_state = self.predicted_states[:, -1]
        final_ref = self.reference_trajectory[:, -1]
        self.cost += self.config.w_fx * (final_state[0] - final_ref[0])**2
        self.cost += self.config.w_fy * (final_state[1] - final_ref[1])**2
        self.cost += self.config.w_ftheta * (final_state[2] - final_ref[2])**2
        self.cost += self.config.w_fv * (final_state[3] - final_ref[3])**2
        
        # 终端铰接角惩罚
        final_phi = final_state[2] - final_state[5]
        self.cost += self.config.w_final_articulation * final_phi**2

        # 设置优化目标
        self.opti.minimize(self.cost)

        # 约束条件
        # 1. 初始状态约束
        self.opti.subject_to(self.predicted_states[:, 0] == self.initial_state)

        # 2. 动力学约束
        for k in range(self.config.N):
            next_state = self.dynamics(self.predicted_states[:, k], self.controls_seq[:, k])
            self.opti.subject_to(self.predicted_states[:, k+1] == next_state)

        # 3. 状态约束
        for k in range(self.config.N + 1):
            # 速度约束
            self.opti.subject_to(self.predicted_states[3, k] >= self.config.v_min)
            self.opti.subject_to(self.predicted_states[3, k] <= self.config.v_max)
            
            # 转向角约束
            self.opti.subject_to(self.predicted_states[4, k] >= self.config.delta_min)
            self.opti.subject_to(self.predicted_states[4, k] <= self.config.delta_max)
            
            # 铰接角约束 
            phi = self.predicted_states[2, k] - self.predicted_states[5, k]
            self.opti.subject_to(phi >= self.config.phi_min)
            self.opti.subject_to(phi <= self.config.phi_max)

        # 4. 控制约束
        for k in range(self.config.N):
            self.opti.subject_to(self.controls_seq[0, k] >= self.config.a_min)
            self.opti.subject_to(self.controls_seq[0, k] <= self.config.a_max)
            self.opti.subject_to(self.controls_seq[1, k] >= -self.config.delta_dot_max)
            self.opti.subject_to(self.controls_seq[1, k] <= self.config.delta_dot_max)

        # 5. 航向角变化率约束 (额外的平滑性约束)
        for k in range(1, self.config.N + 1):
            d_theta = self.predicted_states[2, k] - self.predicted_states[2, k-1]
            self.opti.subject_to(d_theta <= self.config.theta_dot_max * self.config.dt)
            self.opti.subject_to(d_theta >= -self.config.theta_dot_max * self.config.dt)

        # IPOPT求解器设置
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": self.config.max_iter,
            "print_level": 0,  # 减少输出信息
            "acceptable_tol": 1e-4,
            "constr_viol_tol": 1e-4,
            "acceptable_constr_viol_tol": 1e-3
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    def smooth_max(self, z, beta=None):
        """平滑最大函数，用于软约束"""
        if beta is None:
            beta = self.config.obstacle_smooth_beta
        return (1 / beta) * log(1 + exp(beta * z))

    def solve(self, current_state, ref_traj, obstacles):
        """求解MPC优化问题"""
        # 设置优化问题参数
        self.opti.set_value(self.initial_state, current_state)
        self.opti.set_value(self.reference_trajectory, ref_traj)

        # 处理障碍物信息 (预测时域内的位置)
        obs_array = np.zeros((7 * self.config.max_obs, self.config.N + 1))
        for obs_idx, obs in enumerate(obstacles):
            if obs_idx >= self.config.max_obs:
                break
            
            x0, y0, heading, v, length, width, flag = obs
            vx = v * np.cos(heading)  # 障碍物x方向速度
            vy = v * np.sin(heading)  # 障碍物y方向速度
            
            # 预测障碍物在每个时间步的位置
            for k in range(self.config.N + 1):
                t = self.config.dt * k
                obs_array[obs_idx*7 + 0, k] = x0 + vx * t  # x坐标
                obs_array[obs_idx*7 + 1, k] = y0 + vy * t  # y坐标
                obs_array[obs_idx*7 + 2, k] = heading      # 航向角
                obs_array[obs_idx*7 + 3, k] = length       # 长度
                obs_array[obs_idx*7 + 4, k] = width        # 宽度
                obs_array[obs_idx*7 + 6, k] = flag         # 激活标志

        self.opti.set_value(self.obstacles, obs_array)

        # 求解优化问题
        try:
            solution = self.opti.solve()
            optimal_controls = solution.value(self.controls_seq)
            predicted_states = solution.value(self.predicted_states)
            return optimal_controls, predicted_states
        except Exception as e:
            print(f"Semi-trailer MPC求解失败: {e}")
            # 失败时返回紧急制动控制
            safe_controls = np.zeros((2, self.config.N))
            safe_controls[0, :] = self.config.a_min  # 最大制动
            safe_controls[1, :] = 0.0                # 保持转向
            
            # 返回当前状态的简单外推
            safe_states = np.zeros((6, self.config.N + 1))
            for k in range(self.config.N + 1):
                if k == 0:
                    safe_states[:, k] = current_state
                else:
                    # 简单的直线外推 (紧急制动)
                    prev_state = safe_states[:, k-1]
                    new_v = max(0.0, prev_state[3] + self.config.a_min * self.config.dt)
                    safe_states[0, k] = prev_state[0] + new_v * np.cos(prev_state[2]) * self.config.dt
                    safe_states[1, k] = prev_state[1] + new_v * np.sin(prev_state[2]) * self.config.dt
                    safe_states[2, k] = prev_state[2]  # 保持航向
                    safe_states[3, k] = new_v          # 减速
                    safe_states[4, k] = prev_state[4]  # 保持转向
                    safe_states[5, k] = prev_state[5]  # 保持挂车航向
            
            return safe_controls, safe_states