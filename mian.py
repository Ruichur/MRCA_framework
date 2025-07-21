import numpy as np
import scipy.io as sio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from scipy.optimize import least_squares


# =============================================
# Measurement Block:
# =============================================
class MeasurementModel:

    @staticmethod
    def relative_distance(this_state: np.ndarray, that_state: np.ndarray) -> Tuple[np.ndarray, float]:
        dx = that_state[0] - this_state[0]
        dy = that_state[1] - this_state[1]
        d = np.hypot(dx, dy)

        if d < 1e-6:
            return np.zeros((1, 3)), 0.0

        H = np.array([[-dx / d, -dy / d, 0]])

        noise = 0.05
        return H, noise

    @staticmethod
    def relative_angle(this_state: np.ndarray, that_state: np.ndarray) -> Tuple[np.ndarray, float]:
        dx = that_state[0] - this_state[0]
        dy = that_state[1] - this_state[1]
        d_sq = dx ** 2 + dy ** 2

        if d_sq < 1e-6:
            return np.zeros((1, 3)), 0.0

        H = np.array([[-dy / d_sq, dx / d_sq, -1]])

        noise = 0.05
        return H, noise

    @staticmethod
    def normalize_angle(angle):
        angle_mod = angle % (2 * np.pi)
        while angle_mod > np.pi:
            angle_mod -= 2 * np.pi
        while angle_mod < -np.pi:
            angle_mod += 2 * np.pi
        return angle_mod


# =============================================
# Reasoning Block:
# =============================================
class KalmanFilter_Cross_Covariances:
    def __init__(self):
        self.epsilon = 1e-6
        self.delta_t = 0.02
        self.velocity_noise = 0.002
        self.angular_velocity_noise = 0.002

    def predict(self, last_step_state, last_step_covariance, current_step_input):

        state = last_step_state.reshape(3, 1)
        v = current_step_input[0]
        w = current_step_input[1]
        theta = state[2, 0]

        x_new = state[0, 0] + v * np.cos(theta) * self.delta_t
        y_new = state[1, 0] + v * np.sin(theta) * self.delta_t
        theta_new = state[2, 0] + w * self.delta_t
        theta_new = MeasurementModel.normalize_angle(theta_new)
        state_pred = np.array([[x_new], [y_new], [theta_new]])

        F, Q = self._compute_jacobian_and_noise(theta, v)

        covariance_pred = F @ last_step_covariance @ F.T + Q

        return state_pred.reshape(1, 3), covariance_pred

    def _compute_jacobian_and_noise(self, theta, v):

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        delta_t = self.delta_t


        F = np.array([
            [1, 0, -v * sin_theta * delta_t],
            [0, 1, v * cos_theta * delta_t],
            [0, 0, 1]
        ])


        G = np.array([
            [cos_theta * delta_t, 0],
            [sin_theta * delta_t, 0],
            [0, delta_t]
        ])

        V = np.diag([self.velocity_noise ** 2,
                     self.angular_velocity_noise ** 2])

        Q = G @ V @ G.T

        return F, Q

    def update(self, predict_state, predict_covariance, map, map_covariance, measurement, measurement_matrix,
               measurement_noise_cov, robot_id, measurement_id, omega=0.5, cost_func='det'):

        state = predict_state[robot_id - 1, :]
        state = state.reshape(3, 1)
        covariance = predict_covariance[robot_id - 1]
        measurement = measurement.reshape(2, 1)

        innovation_cov = np.dot(measurement_matrix, np.dot(covariance, measurement_matrix.T)) + measurement_noise_cov
        innovation_cov += np.eye(innovation_cov.shape[0]) * 1e-6

        kalman_gain = np.dot(np.dot(covariance, measurement_matrix.T), np.linalg.inv(innovation_cov))

        innovation = measurement - np.dot(measurement_matrix, state)
        state = state + np.dot(kalman_gain, innovation)
        x , y, theta = state
        theta = MeasurementModel.normalize_angle(theta)
        state[2, 0] = theta
        state = state.flatten()

        covariance = covariance - np.dot(kalman_gain, np.dot(measurement_matrix, covariance))

        if 6 <= measurement_id <= 20:
            other_state = map[measurement_id - 6, :]
            other_cov = map_covariance[measurement_id - 6]
        else:
            other_state = predict_state[measurement_id - 1, :]
            other_cov = predict_covariance[measurement_id - 1]

        if omega is None:
            current_omega = self.optimize_omega(state, covariance, other_state, other_cov, cost_func)
        else:
            current_omega = omega

        state, covariance = self.fuse_with_ci(state, covariance, other_state, other_cov, current_omega)

        return state, covariance

    def fuse_with_ci(self, state1, cov1, state2, cov2, omega):

        inv_cov1 = np.linalg.inv(cov1)
        inv_cov2 = np.linalg.inv(cov2)
        combined_inv_cov = omega * inv_cov1 + (1 - omega) * inv_cov2
        combined_cov = np.linalg.inv(combined_inv_cov)
        combined_state = combined_cov @ (
                    omega * inv_cov1 @ state1.reshape(-1, 1) + (1 - omega) * inv_cov2 @ state2.reshape(-1, 1))

        return combined_state.flatten(), combined_cov

    def optimize_omega(self, state1, cov1, state2, cov2, cost_func):
        omegas = np.linspace(0, 1, 100)
        best_omega, min_cost = 0.5, np.inf
        for omega in omegas:
            _, fused_cov = self.fuse_with_ci(state1, cov1, state2, cov2, omega)
            if cost_func == 'det':
                cost = np.linalg.det(fused_cov)
            elif cost_func == 'trace':
                cost = np.trace(fused_cov)
            if cost < min_cost:
                min_cost, best_omega = cost, omega
        return best_omega


# =============================================
# Cognition Block:
# =============================================

def residual(theta, positions, orientations, distances, angles):
    x, y = theta
    residuals = []
    for i in range(len(positions)):
        pred_distance = np.sqrt((x - positions[i, 0]) ** 2 + (y - positions[i, 1]) ** 2)

        dx = x - positions[i, 0]
        dy = y - positions[i, 1]
        pred_global_angle = np.arctan2(dy, dx)
        pred_relative_angle = pred_global_angle - orientations[i]
        pred_relative_angle = MeasurementModel.normalize_angle(pred_relative_angle)

        residuals.append(pred_distance - distances[i])
        residuals.append(pred_relative_angle - angles[i])

    return np.array(residuals)

@dataclass
class AnchorBuffer:
    robot_states: List[np.ndarray] = field(default_factory=list) 
    robot_cov: List[np.ndarray] = field(default_factory=list) 
    measurement: List[np.ndarray] = field(default_factory=list)
    count: int = 0 
    last_estimate: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan, 0]))
    last_cov: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan, 0]))
    initialized: List[bool] = field(default_factory=lambda: [False] * 15)


class CooperativeCognition:

    def __init__(self, num_robots: int, num_anchors: int):
        self.num_robots = num_robots
        self.num_anchors = num_anchors
        self.initial_guess = np.array([0, 0])

        self.anchor_buffers = {aid: AnchorBuffer() for aid in range(6, 21)}

        self.map_global = np.full((num_anchors, 3), np.nan)
        self.initial_state_all = np.full((num_anchors, 3), np.nan)
        self.local_maps = [np.full((num_anchors, 3), np.nan) for _ in range(num_robots)]
        self.map_covariances = [[np.eye(3) for _ in range(num_anchors)] for _ in range(num_robots)]

    def add_measurement_to_buffer(self, robot_id: int, anchor_id: int,
                                  robot_state: np.ndarray, measurement: np.ndarray, robot_cov: np.ndarray):

        buffer = self.anchor_buffers[anchor_id]

        if np.any(np.isnan(self.local_maps[robot_id - 1][anchor_id - 6])):
            if buffer.count >= buffer_num:
                buffer.robot_states.pop(0)
                buffer.measurement.pop(0)
                buffer.robot_cov.pop(0)
            else:
                buffer.count += 1
            buffer.robot_states.append(robot_state.copy())
            buffer.measurement.append(measurement.copy())
            buffer.robot_cov.append(robot_cov.copy())
            if buffer.count == buffer_num and not buffer.initialized[anchor_id - 6]:
                self.perform_initialization(robot_id, anchor_id)

    def perform_initialization(self, robot_id: int, anchor_id: int):
        anchor_idx = anchor_id - 6
        buffer = self.anchor_buffers[anchor_id]

        positions = np.array([array[:2] for array in buffer.robot_states])
        orientations = np.array([array[2] for array in buffer.robot_states])
        distances = np.array([array[0] for array in buffer.measurement])
        angles = np.array([array[1] for array in buffer.measurement])

        initial_guess = np.array([0, 0])

        result = least_squares(
            residual,
            initial_guess,
            args=(positions, orientations, distances, angles),
            method='lm',
            max_nfev=50
        )

        x, y = result.x
        initial_state = np.array([x, y, 0])

        d_est = np.sqrt(((initial_state[0:2] - positions) ** 2).sum(axis=1))
        A = np.array([(initial_state[0] - positions[:, 0]) / d_est,
                      (initial_state[1] - positions[:, 1]) / d_est]).T

        C = np.linalg.inv(A.T @ A + np.eye(2) * 1e-6)
        gdop = np.sqrt(np.trace(C))

        if gdop > 1.5:
            return

        anchor_cov = np.sum(np.stack(buffer.robot_cov), axis=0) / buffer_num

        buffer.initialized[anchor_id - 6] = True

        self.local_maps[robot_id - 1][anchor_idx] = initial_state
        self.map_covariances[robot_id - 1][anchor_idx] = anchor_cov
        self.initial_state_all[anchor_idx] = initial_state
        self.map_global[anchor_idx] = initial_state

    def synchronize_maps(self, robot_id, neighbor_id):
        ridx = robot_id - 1
        nidx = neighbor_id - 1

        for anchor_idx in range(self.num_anchors):
            cov_rid = self.map_covariances[ridx][anchor_idx]
            cov_nid = self.map_covariances[nidx][anchor_idx]
            anchor_nidx = self.local_maps[nidx][anchor_idx]
            anchor_ridx = self.local_maps[ridx][anchor_idx]

            if np.isnan(anchor_nidx).any() and np.isnan(anchor_ridx).any():
                continue

            elif np.isnan(anchor_ridx).any():
                best_state = anchor_nidx
                best_cov = cov_nid

            elif np.isnan(anchor_nidx).any():
                best_state = anchor_ridx
                best_cov = cov_rid

            else:
                trace_rid = np.trace(cov_rid)
                trace_nid = np.trace(cov_nid)

                if trace_rid <= trace_nid:
                    best_state = anchor_ridx
                    best_cov = cov_rid
                else:
                    best_state = anchor_nidx
                    best_cov = cov_nid

            self.local_maps[ridx][anchor_idx] = best_state
            self.local_maps[nidx][anchor_idx] = best_state
            self.map_covariances[ridx][anchor_idx] = best_cov
            self.map_covariances[nidx][anchor_idx] = best_cov
            self.map_global[anchor_idx] = best_state

# =============================================
# Application Block:
# =============================================
@dataclass
class RobotData:
    groundtruth: Dict[float, np.ndarray] = field(default_factory=dict)
    measurement: Dict[float, np.ndarray] = field(default_factory=dict)
    odometry: Dict[float, np.ndarray] = field(default_factory=dict)

def root_mean_square_error(estimated_states, true_states):
    errors = estimated_states - true_states
    return np.sqrt(np.mean((errors) ** 2))

class MRCAFramework:

    def __init__(self, data_folder: str, num_robots: int = 5, num_anchors: int = 15):  # 添加参数
        self.num_robots = num_robots
        self.num_anchors = num_anchors
        self.sample_time = 0.02

        self.measurement = MeasurementModel()
        self.reasoning = KalmanFilter_Cross_Covariances()
        self.cognition = CooperativeCognition(num_robots, num_anchors)  # 传递参数

        self.robot_data = self._load_robot_data(data_folder)
        self.true_map = self._load_map_data(data_folder)

        self.robot_states = np.zeros((num_robots, 3))
        self.robot_covs = [np.eye(3) for _ in range(num_robots)]
        self.trajectories = {rid: [] for rid in range(1, num_robots + 1)}

        self.velocity_biases = np.random.normal(0, 0.05, num_robots)
        self.omega_biases = np.random.normal(0, 0.01, num_robots)

        self._initialize_robot_states()

    def _initialize_robot_states(self):
        for rid in range(1, self.num_robots + 1):
            if self.robot_data[rid].groundtruth:
                first_timestamp = min(self.robot_data[rid].groundtruth.keys())
                self.robot_states[rid - 1] = self.robot_data[rid].groundtruth[first_timestamp][:3]

    def _load_robot_data(self, data_folder: str) -> Dict[int, RobotData]:
        robot_data = {}
        for rid in range(1, 6):
            gt_file = os.path.join(data_folder, f"Robot{rid}_Groundtruth.mat")
            meas_file = os.path.join(data_folder, f"Robot{rid}_Measurement.mat")
            odo_file = os.path.join(data_folder, f"Robot{rid}_Odometry.mat")

            robot = RobotData(
                groundtruth=self._load_mat(gt_file, f"Robot{rid}_Groundtruth"),
                measurement=self._load_mat(meas_file, f"Robot{rid}_Measurement"),
                odometry=self._load_mat(odo_file, f"Robot{rid}_Odometry")
            )
            robot_data[rid] = robot
        return robot_data

    def _load_map_data(self, data_folder: str) -> np.ndarray:
        map_file = os.path.join(data_folder, "Landmark_Groundtruth.mat")
        return sio.loadmat(map_file)["Landmark_Groundtruth"][:, 1:3]

    def _load_mat(self, file_path: str, var_name: str) -> Dict[float, np.ndarray]:
        data = sio.loadmat(file_path)[var_name]
        return {round(row[0], 2): row[1:] for row in data if row.size > 1}

    def run(self):

        timestamps = sorted(self.robot_data[1].odometry.keys())
        step = 0

        for t in timestamps:
            step += 1
            for rid in range(1, self.num_robots + 1):

                if t in self.robot_data[rid].odometry:

                    u = self.robot_data[rid].odometry[t]

                    current_state = self.robot_states[rid - 1].copy()

                    new_state, new_cov = self.reasoning.predict(
                        last_step_state=current_state,
                        last_step_covariance=self.robot_covs[rid - 1],
                        current_step_input=u
                    )

                    self.robot_states[rid - 1] = new_state.flatten()
                    self.robot_covs[rid - 1] = new_cov

            for rid in range(1, self.num_robots + 1):
                if t in self.robot_data[rid].measurement:
                    meas_data = self.robot_data[rid].measurement[t]
                    if meas_data.size > 0:
                        meas_id = int(meas_data[0])
                        measurement = meas_data[1:3]

                        if 6 <= meas_id <= 20:
                            anchor_idx = meas_id - 6

                            self.cognition.add_measurement_to_buffer(
                                rid, meas_id,
                                self.robot_states[rid - 1],
                                measurement,
                                self.robot_covs[rid - 1]
                            )

                            if not np.isnan(self.cognition.local_maps[rid - 1][anchor_idx]).any():
                                target_state = self.cognition.local_maps[rid - 1][anchor_idx]
                            else:
                                continue
                        else:
                            target_state = self.robot_states[meas_id - 1]
                            if step % 50.0 == 0:
                                self.cognition.synchronize_maps(rid, meas_id)

                        H_dist, R_dist = self.measurement.relative_distance(
                            self.robot_states[rid - 1], target_state
                        )
                        H_angle, R_angle = self.measurement.relative_angle(
                            self.robot_states[rid - 1], target_state
                        )

                        measurement_matrix = np.vstack([H_dist, H_angle])
                        measurement_noise_cov = np.diag([R_dist, R_angle])

                        self.robot_states[rid - 1], self.robot_covs[rid - 1] = \
                            self.reasoning.update(
                                predict_state=self.robot_states,
                                predict_covariance=self.robot_covs,
                                map=self.cognition.local_maps[rid - 1],
                                map_covariance=self.cognition.map_covariances[rid - 1],
                                measurement=measurement,
                                measurement_matrix=measurement_matrix,
                                measurement_noise_cov=measurement_noise_cov,
                                robot_id=rid,
                                measurement_id=meas_id
                            )

        errs_last_t = []
        for rid in range(1, 6):
            est_xy = self.robot_states[rid - 1, :2]

            true_xy = np.array(self.robot_data[rid].groundtruth[t])[:2]

            errs_last_t.append(np.linalg.norm(est_xy - true_xy))

        print("CUMULATIVE POSITION RMSE:", np.mean(errs_last_t))

if __name__ == "__main__":
    # 配置参数
    buffer_num = 15
    DATA_FOLDER = r"\DATA_output7"
    system = MRCAFramework(DATA_FOLDER)
    system.run()
