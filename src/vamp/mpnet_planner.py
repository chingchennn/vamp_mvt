import numpy as np
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import logging
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MAX_POINTCLOUD_SIZE = 11978
ROBOT_DOF = 8

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MAX_POINTCLOUD_SIZE*3, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 28)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class Planner(nn.Module):
    def __init__(self, input_size, output_size):
        super(Planner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
            nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
            nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
            nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
            nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out

class HardwareBackend(ABC):
    """Hardware abstraction for neural network inference"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        pass
    
    @abstractmethod
    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        pass

class CPUBackend(HardwareBackend):
    """CPU inference backend"""
    
    def load_model(self, model_path: str) -> Any:
        path = Path(model_path)
        if path.suffix == '.pkl':
            import pickle
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                # If pickle fails, try PyTorch state_dict
                import torch
                import os
                state_dict = torch.load(path, map_location='cpu')
                filename = os.path.basename(path)
                if "encoder" in filename.lower():
                    model = Encoder()
                    model.load_state_dict(state_dict)
                elif "planner" in filename.lower():
                    model = Planner(28 + ROBOT_DOF + ROBOT_DOF, ROBOT_DOF)  # latent + start + goal -> next_config
                    model.load_state_dict(state_dict)
                model.eval()
                return model
        elif path.suffix in ['.pt', '.pth']:
            import torch
            model = torch.load(path, map_location='cpu')
            model.eval()
            return model
        elif path.suffix == '.onnx':
            import onnxruntime as ort
            return ort.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict'):  # sklearn/pickle
            return model.predict(inputs)
        elif hasattr(model, '__call__'):  # pytorch
            import torch
            with torch.no_grad():
                tensor_input = torch.from_numpy(inputs).float()
                if len(tensor_input.shape) == 1:
                    tensor_input = tensor_input.unsqueeze(0)
                output = model(tensor_input)
                return output.numpy()
        elif hasattr(model, 'run'):  # onnx
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: inputs})
            return outputs[0]
        else:
            raise ValueError("Unknown model type")

class OrangePiNPUBackend(HardwareBackend):
    """OrangePi NPU backend with CPU fallback"""
    
    def __init__(self):
        self.cpu_backend = CPUBackend()
        logger.info("OrangePi NPU: Using CPU fallback (NPU implementation pending)")
    
    def load_model(self, model_path: str) -> Any:
        # TODO: Replace with Raspberry Pi NPU model loading
        return self.cpu_backend.load_model(model_path)
    
    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        # TODO: Replace with OrangePi NPU inference
        return self.cpu_backend.predict(model, inputs)

class RaspberryPiNPUBackend(HardwareBackend):
    """Raspberry Pi NPU backend with CPU fallback"""
    
    def __init__(self):
        self.cpu_backend = CPUBackend()
        logger.info("Raspberry Pi NPU: Using CPU fallback (NPU implementation pending)")
    
    def load_model(self, model_path: str) -> Any:
        # TODO: Replace with Raspberry Pi NPU model loading
        return self.cpu_backend.load_model(model_path)
    
    def predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        # TODO: Replace with Raspberry Pi NPU inference
        return self.cpu_backend.predict(model, inputs)

class MPNetPlanner:
    """Compact MPNet planner using Python orchestration"""
    
    BACKENDS = {
        'cpu': CPUBackend,
        'opi_npu': OrangePiNPUBackend,
        'rpi_npu': RaspberryPiNPUBackend,
    }
    
    PNET_INFERENCE_CNT = 0

    def __init__(self, encoder_path: str, planner_path: str, hardware: str = 'cpu'):
        """Initialize MPNet with specified hardware backend"""
        if hardware not in self.BACKENDS:
            logger.warning(f"Unknown hardware '{hardware}', using CPU")
            hardware = 'cpu'
        
        self.backend = self.BACKENDS[hardware]()
        self.hardware = hardware
        
        # Load models
        logger.info(f"Loading encoder: {encoder_path}")
        self.encoder = self.backend.load_model(encoder_path)
        logger.info(f"Loading planner: {planner_path}")
        self.planner = self.backend.load_model(planner_path)
        
        # Cache for encoded environment
        self.latent_code = None
        logger.info(f"MPNet initialized with {hardware} backend")
    
    def encode_environment(self, pointcloud: list) -> bool:
        """Encode point cloud to latent representation"""
        try:
            # Preprocess if needed (normalize, filter, etc.)
            logger.debug(f"Point cloud shape before preprocessing: {np.array(pointcloud).shape}")
            processed_cloud = self._preprocess_pointcloud(pointcloud)
            logger.debug(f"Point cloud shape after preprocessing: {np.array(processed_cloud).shape}")
            
            # Encode
            self.latent_code = self.backend.predict(self.encoder, processed_cloud)
            logger.debug(f"Encoded {len(pointcloud)} points to latent shape: {self.latent_code.shape}")
            return True
        except Exception as e:
            logger.error(f"Environment encoding failed: {e}")
            return False
    
    def plan(self, start: np.ndarray, goal: np.ndarray, robot_module, environment,
             max_iterations: int = 50, max_planning_steps: int = 50) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal using MPNet
        
        Args:
            start: Start configuration
            goal: Goal configuration  
            robot_module: VAMP robot module (e.g., vamp.panda)
            environment: VAMP environment
            max_iterations: Max planning iterations
            max_planning_steps: Max steps per planning attempt
            
        Returns:
            List of configurations forming path, or None if failed
        """
        if self.latent_code is None:
            logger.error("Environment not encoded. Call encode_environment() first.")
            return None
        
        print(f"Start shape: {len(start)}, Goal shape: {len(goal)}")
        
        start_time = time.time()
        
        # Check direct connection first
        if robot_module.validate_motion(robot_module.Configuration(start), robot_module.Configuration(goal), environment):
            logger.info("Direct connection found")
            return [start, goal]
        
        best_path = None
        best_distance_to_goal = float('inf')
        
        for iteration in range(max_iterations):
            # Try bidirectional planning (core MPNet algorithm)
            path = self._bidirectional_planning_attempt(
                start, goal, robot_module, environment, max_planning_steps
            )
            # bookshelf_thin, fetch, problem index 3
            # path = [[0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
            #         [0.1130716415169656, 0.6701656662448738, 0.5437118304124002, -0.9573185442113827, 1.192729119155159, -0.6570224510957486, 1.389650191854703, 0.5608505081960977],
            #         [0.09654528291702527, 0.5147561598239914, 0.4493044145424446, -0.4633838174423112, 0.1591385738381093, -0.8442762895832869, 0.5673636381620717, -0.6596020426390228],
            #         [0.08840026260867515, -0.1049776956095018, 0.2490675657194144, -0.255429554351454, -1.351346779291686, -1.259964889635087, -0.2211360368887943, -2.91253456987987],
            #         [0.06580220830306079, -1.002794560663853, -0.272744465625363, -1.431699601650493, -1.561234119725792, -1.744681742687049, -0.04365573011807782, 1.117031525941965],
            #         [0.2162197405043042, -0.2457764831227006, 0.02184441132129261, -2.045275807819388, -0.009591529666930394, 0.3332992301497282, 0.1652488400061966, 1.713909096307712]]
            if path and len(path) > 1:
                # Check if we reached goal
                final_config = path[-1]
                logger.debug(f"Check distance between last config {final_config} and goal {goal}")
                distance_to_goal = np.linalg.norm(final_config - goal)
                if distance_to_goal < 1.0:
                    path.append(goal)
                    logger.info(f"Path found in iteration {iteration + 1}: {len(path)} waypoints")
                    logger.info(f"Planning time: {(time.time() - start_time) * 1000:.1f}ms")
                    logger.info(f"Planning Network Inference Count: {self.PNET_INFERENCE_CNT}")
                    return path
                
                # Track best partial path
                if distance_to_goal < best_distance_to_goal:
                    best_distance_to_goal = distance_to_goal
                    best_path = path.copy()
        
        if best_path:
            logger.warning(f"Partial path found (distance to goal: {best_distance_to_goal:.3f})")
            return best_path
        
        logger.info("No path found")
        return None
    
    def _bidirectional_planning_attempt(self, start: np.ndarray, goal: np.ndarray,
                                      robot_module, environment, max_steps: int) -> Optional[List[np.ndarray]]:
        """MPNet bidirectional planning with divide-and-conquer strategy"""
        
        # Forward planning: start -> intermediate
        forward_path = self._single_planning_attempt(start, goal, robot_module, environment, max_steps // 2)
        if not forward_path or len(forward_path) < 2:
            return None
        
        # Backward planning: goal -> intermediate  
        backward_path = self._single_planning_attempt(goal, start, robot_module, environment, max_steps // 2)
        if not backward_path or len(backward_path) < 2:
            return forward_path
        
        # Try to connect forward and backward paths
        forward_end = forward_path[-1]
        backward_end = backward_path[-1]
        
        # Check if paths can be connected
        if robot_module.validate_motion(robot_module.Configuration(forward_end), robot_module.Configuration(backward_end), environment):
            # Merge paths (reverse backward path since it was planned goal->start)
            merged_path = forward_path + list(reversed(backward_path[:-1]))
            return merged_path
        
        # If direct connection fails, try recursive planning between end points
        bridge_path = self._single_planning_attempt(forward_end, backward_end, robot_module, environment, max_steps // 4)
        if bridge_path and len(bridge_path) > 1:
            merged_path = forward_path + bridge_path[1:] + list(reversed(backward_path[:-1]))
            return merged_path
        
        # Return the longer path if connection fails
        return forward_path if len(forward_path) >= len(backward_path) else backward_path
    
    def _single_planning_attempt(self, start: np.ndarray, goal: np.ndarray,
                               robot_module, environment, max_steps: int) -> Optional[List[np.ndarray]]:
        """Single direction MPNet planning attempt"""
        current = start.copy()
        path = [current.copy()]
        
        for step in range(max_steps):
            # Direct neural prediction from current to goal
            next_config = self._predict_next_config(current, goal)
            
            if next_config is None:
                break
            
            # Validate motion
            if robot_module.validate_motion(robot_module.Configuration(current), robot_module.Configuration(next_config), environment):
                path.append(next_config.copy())
                current = next_config
                
                # Check if reached goal
                if np.linalg.norm(current - goal) < 1:  # Goal tolerance
                    return path
            else:
                # Collision detected - try to recover with slight perturbation
                perturbed_config = self._add_noise_to_config(next_config, noise_scale=0.25)
                if robot_module.validate_motion(robot_module.Configuration(current), robot_module.Configuration(perturbed_config), environment):
                    logger.debug(f"Recovered with perturbed config: {perturbed_config}")
                    path.append(perturbed_config.copy())
                    current = perturbed_config
                else:
                    # Failed to recover, stop this planning attempt
                    break
        
        return path if len(path) > 1 else None
    
    def _predict_next_config(self, current: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
        """Predict next configuration using planning network"""
        try:
            # Prepare input: [latent_code, current, goal]
            planning_input = self._prepare_planning_input(current, goal)
            
            # Predict
            prediction = self.backend.predict(self.planner, planning_input)
            self.PNET_INFERENCE_CNT = self.PNET_INFERENCE_CNT + 1
            
            # Post-process prediction
            return self._postprocess_prediction(prediction, current)
        except Exception as e:
            logger.debug(f"Prediction failed: {e}")
            return None
    
    def _preprocess_pointcloud(self, pointcloud: list) -> np.ndarray:
        """Preprocess pointcloud for encoder"""
        pointcloud = np.array(pointcloud)
        
        # Handle different input formats
        if pointcloud.ndim == 1:
            # Already flattened
            if len(pointcloud) == MAX_POINTCLOUD_SIZE * 3:
                return pointcloud.astype(np.float32)
            else:
                # Reshape and process
                pointcloud = pointcloud.reshape(-1, 3)
        
        # Subsample if too many points
        if len(pointcloud) > MAX_POINTCLOUD_SIZE:
            indices = np.random.choice(len(pointcloud), MAX_POINTCLOUD_SIZE, replace=False)
            pointcloud = pointcloud[indices]
        
        # Pad if too few points
        elif len(pointcloud) < MAX_POINTCLOUD_SIZE:
            padding = np.zeros((MAX_POINTCLOUD_SIZE - len(pointcloud), 3))
            pointcloud = np.vstack([pointcloud, padding])
        
        # # Normalize point cloud (center around origin)
        # if len(pointcloud) > 0:
        #     centroid = np.mean(pointcloud, axis=0)
        #     pointcloud = pointcloud - centroid
            
        return pointcloud.astype(np.float32).flatten()
    
    def _prepare_planning_input(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Prepare input for planning network"""
        # Flatten latent code if needed
        latent_flat = self.latent_code.flatten() if self.latent_code.ndim > 1 else self.latent_code
        
        # Concatenate: [latent_code, current_config, goal_config]
        planning_input = np.concatenate([latent_flat, current, goal])
        
        # Add batch dimension if needed
        if planning_input.ndim == 1:
            planning_input = planning_input.reshape(1, -1)
        
        return planning_input.astype(np.float32)
    
    def _postprocess_prediction(self, prediction: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Post-process network prediction"""
        # Remove batch dimension if present
        if prediction.ndim > 1:
            prediction = prediction.flatten()
        
        # Apply movement constraints (limit step size for stability)
        max_step_size = 0.3  # radians
        step = prediction - current
        step_norm = np.linalg.norm(step)
        
        if step_norm > max_step_size:
            step = step * (max_step_size / step_norm)
            prediction = current + step
        
        return prediction
    
    def _add_noise_to_config(self, config: np.ndarray, noise_scale: float = 0.05) -> np.ndarray:
        """Add small random noise to configuration for collision recovery"""
        noise = np.random.normal(0, noise_scale, config.shape)
        noisy_config = config + noise
        return np.clip(noisy_config, -np.pi, np.pi)

def plan_with_mpnet(robot_name: str, start: List[float], goal: List[float],
                   environment, pointcloud, encoder_path: str, planner_path: str,
                   hardware: str = 'cpu') -> Optional[List[np.ndarray]]:
    """
    MPNet planning
    
    Args:
        robot_name: Robot name ('panda', 'ur5', 'fetch', etc.)
        start: Start configuration as list
        goal: Goal configuration as list
        environment: VAMP environment
        pointcloud: Point cloud as ndarray
        encoder_path: Path to encoder model
        planner_path: Path to planner model
        hardware: Hardware backend ('cpu', 'opi_npu', 'rpi_npu')
        
    Returns:
        Path as list of configurations, or None if planning failed
    """
    import vamp
    
    # Get robot module
    robot_module = getattr(vamp, robot_name)
    
    # Initialize MPNet
    mpnet = MPNetPlanner(encoder_path, planner_path, hardware)
    
    # Extract and encode environment
    if not mpnet.encode_environment(pointcloud):
        logger.error("Failed to encode environment")
        return None
    
    # Convert to numpy arrays
    start_np = np.array(start, dtype=np.float32)
    goal_np = np.array(goal, dtype=np.float32)
    
    # Plan
    path = mpnet.plan(start_np, goal_np, robot_module, environment)
    
    return path