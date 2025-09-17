import numpy as np

class SemiTrailerConfig:
    def __init__(self):
        # -------------------------- MPC核心参数 --------------------------
        self.N = 20               # 更长预测时域 (was 15)
        self.dt = 0.2            
        self.max_obs = 5         

        # -------------------------- 半挂车物理参数 --------------------------
        # CHANGED: Separate tractor and trailer parameters
        self.L_t = 3.0               # 牵引车轴距 [m] (was self.L = 1.5)
        self.L_s = 12.0              # 挂车长度 [m] (NEW)
        self.tractor_length = 6.0    # 牵引车总长 [m] (was vehicle_length = 2.0)
        self.trailer_length = 12.0   # 挂车长度 [m] (NEW)
        self.vehicle_width = 2.5     # 车辆宽度 [m] (was 1.0)
        self.total_length = 18.0     # 总长度 [m] (NEW)
        self.safety_distance = 1.0   # 更大安全距离 [m] (was 0.0)

        # -------------------------- 车辆运动约束 --------------------------
        self.v_max = 25.0          # 高速公路最大速度 [m/s] (was 10.0)
        self.v_min = 0.0           
        self.a_max = 1.5           # 更保守的加速度 [m/s²] (was 2.0)
        self.a_min = -4.0          # 更好的制动能力 [m/s²] (was -3.0)
        self.delta_max = np.deg2rad(25)  # 更有限的转向角 [rad] (was 30)
        self.delta_min = -np.deg2rad(25) 
        self.delta_dot_max = np.deg2rad(10)  # 更慢的转向速率 [rad/s] (was 20)
        self.theta_dot_max = np.deg2rad(30)  # [rad/s] (was 45)

        # -------------------------- NEW: 铰接约束 --------------------------
        self.phi_max = np.deg2rad(60)     # 最大铰接角 [rad] (防止折刀)
        self.phi_min = -np.deg2rad(60)    # 最小铰接角 [rad]

        # -------------------------- 道路与场景参数 --------------------------
        self.road_length = 50.0    # 更长的道路 [m] (was 20.0)
        self.y_min = -4.0          # 更宽的道路边界 [m] (was -2.5)
        self.y_max = 4.0           # [m] (was 2.5)
        # CHANGED: 6 states now [x, y, θ_tractor, v, δ, θ_trailer]
        self.start_pos = np.array([2.0, 0.0, 0.0, 15.0, 0.0, 0.0])  # (was 4 states)
        self.goal_pos = np.array([45.0, 0.0, 0.0, 15.0, 0.0, 0.0])  # (was 4 states)

        # -------------------------- 可视化参数 --------------------------
        self.xlim = (-2, self.road_length + 2)  
        self.ylim = (self.y_min - 1, self.y_max + 1)  
        self.animation_interval = 100  

        # -------------------------- 成本函数权重 --------------------------
        self.w_x = 1.0          
        self.w_y = 2.0          # 更重要的横向精度 (was 1.0)
        self.w_theta = 1.0      
        self.w_v = 0.1          
        self.w_a = 0.1          
        self.w_delta_dot = 0.1  
        self.w_bound = 100.0    # 更高的边界惩罚 (was 50.0)
        self.w_obstacle = 30.0  # 更高的障碍物惩罚 (was 20.0)
        
        # NEW: 铰接角惩罚权重
        self.w_articulation = 5.0        # 铰接角惩罚
        self.w_final_articulation = 10.0 # 终端铰接角惩罚
        
        self.w_fx = 10.0        
        self.w_fy = 15.0        # 更重要的终端横向误差 (was 10.0)
        self.w_ftheta = 10.0    
        self.w_fv = 0.5         

        # 障碍物惩罚参数
        self.obstacle_decay_rate = 3.0   # 更缓和的衰减 (was 4.0)
        self.obstacle_smooth_beta = 4.0  # 更软的约束 (was 6.0)

        # -------------------------- 仿真参数 --------------------------
        self.sim_time = 40.0    # 更长的仿真时间 [s] (was 30.0)
        self.ref_speed = 20.0   # 高速公路参考速度 [m/s] (was 1.0)
        self.goal_threshold = 1.0  # 更大的容差 [m] (was 0.2)
        self.max_iter = 300     # 更多迭代次数 (was 200)