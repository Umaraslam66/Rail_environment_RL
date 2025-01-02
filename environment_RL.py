import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Configuration class for train and network parameters."""
    beta_waiting: float = 0.7  # Recovery rate for waiting time
    beta_running: float = 0.7  # Recovery rate for running time
    constraint_violation_penalty: float = 1.0  # Penalty for constraint violations
    action_weight: float = 0.5  # Weight for action penalties
    delay_weight: float = 1.0  # Weight for delay penalties
    conflict_weight: float = 1.0  # Weight for conflict penalties
    exploration_weight: float = 0.05  # Weight for exploration bonus
    headway_time: float = 0.0013889  # Minimum headway time between trains (normalized)
    max_runtime_factor: float = 2.5  # Maximum allowed runtime as factor of minimum runtime
    max_dwelltime_factor: float = 2.5  # Maximum allowed dwell time as factor of minimum dwell time

class TrainNetworkEnv(gym.Env):
    """
    A generic train network simulation environment for reinforcement learning.
    
    This environment simulates train operations on a railway network, handling:
    - Train movement between stations
    - Dwell times at stations
    - Delays and recovery
    - Train conflicts and overtaking
    - Schedule adherence
    
    The agent controls train operations by deciding:
    - Running times between stations (speed control)
    - Dwell times at stations (station stops)
    
    States include:
    - Current station position
    - Time information
    - Nearby train positions
    - Schedule information
    
    Actions:
    - Continuous values for running time and dwell time adjustments
    """
    
    def __init__(
        self,
        stations: List[str],
        runtimes: Dict[str, float],
        dwelltimes: Dict[str, float],
        base_schedule: Dict[str, List[Tuple[int, float, float, int]]],
        config: Optional[TrainConfig] = None,
        max_nearby_trains: int = 4,
        max_steps: int = 20
    ):
        """
        Initialize the train network environment.
        
        Args:
            stations: List of station names in order
            runtimes: Dictionary of minimum running times between stations
            dwelltimes: Dictionary of minimum dwell times at stations
            base_schedule: Reference timetable with train IDs, arrival times, departure times, and directions
            config: Configuration parameters for the environment
            max_nearby_trains: Maximum number of nearby trains to track
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        self.config = config or TrainConfig()
        self.stations = stations
        self.runtimes = runtimes
        self.dwelltimes = dwelltimes
        self.base_schedule = base_schedule
        self.max_nearby_trains = max_nearby_trains
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Calculate observation space size
        obs_size = (
            len(self.stations) +  # One-hot encoding of current station
            1 +  # Current time
            self.max_nearby_trains * 4 +  # Nearby train positions
            2 +  # Departure window
            1 +  # Violation flag
            1  # Cumulative travel time
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0] * obs_size),
            high=np.array([1] * obs_size),
            dtype=np.float64
        )
        
        # Initialize state variables
        self.reset_state_variables()

    def reset_state_variables(self):
        """Reset all state variables to their initial values."""
        self.current_station_index = 0
        self.current_time = 0.0
        self.cumulative_reward = 0.0
        self.steps_taken = 0
        self.violation_occurred = False
        self.tracked_trains = {'preceding': [], 'following': []}
        self.times = {
            'arrival': [0.0],
            'departure': [0.0],
            'actual_arrival': [0.0],
            'actual_departure': [0.0]
        }

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        self.reset_state_variables()
        
        # Set initial departure time
        self.current_time = options.get('initial_time', 0.44)
        
        # Initialize schedules
        self.current_schedule = self._generate_modified_schedule()
        
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an environment step based on the agent's action.
        
        Args:
            action: Array of [dwell_time_factor, runtime_factor]
            
        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether episode has ended
            truncated: Whether episode was artificially terminated
            info: Additional information
        """
        if self.steps_taken >= self.max_steps:
            return None, self.cumulative_reward, True, True, {}
            
        # Save current state
        state_snapshot = self._save_state()
        
        # Process action
        is_feasible, reward, info = self._process_action(action)
        
        if not is_feasible:
            self._restore_state(state_snapshot)
            return self._get_observation(), reward, False, False, info
            
        # Update state
        self.cumulative_reward += reward
        self.steps_taken += 1
        self.current_station_index += 1
        
        # Check termination
        done = self.current_station_index >= len(self.stations)
        
        return self._get_observation(), reward, done, False, info

    def _process_action(
        self, 
        action: np.ndarray
    ) -> Tuple[bool, float, Dict]:
        """
        Process the agent's action and calculate outcomes.
        
        Args:
            action: Array of [dwell_time_factor, runtime_factor]
            
        Returns:
            is_feasible: Whether the action results in a feasible state
            reward: Reward for the action
            info: Additional information
        """
        dwell_action, runtime_action = action
        
        # Calculate times
        runtime = self._calculate_runtime(runtime_action)
        dwelltime = self._calculate_dwelltime(dwell_action)
        
        # Calculate arrivals and departures
        arrival_time = self.times['departure'][-1] + runtime
        departure_time = arrival_time + dwelltime
        
        # Check feasibility
        is_feasible = self._check_feasibility(arrival_time, departure_time)
        
        if is_feasible:
            # Update times
            self._update_times(arrival_time, departure_time)
            
            # Calculate reward
            reward = self._calculate_reward(action, arrival_time, departure_time)
        else:
            reward = -self.config.constraint_violation_penalty
            
        return is_feasible, reward, {'feasible': is_feasible}

    def _calculate_runtime(self, runtime_action: float) -> float:
        """Calculate actual runtime based on action."""
        base_runtime = self.runtimes[self.stations[self.current_station_index]]
        max_runtime = base_runtime * self.config.max_runtime_factor
        return runtime_action * base_runtime + (1 - runtime_action) * max_runtime

    def _calculate_dwelltime(self, dwell_action: float) -> float:
        """Calculate actual dwell time based on action."""
        base_dwelltime = self.dwelltimes[self.stations[self.current_station_index]]
        max_dwelltime = base_dwelltime * self.config.max_dwelltime_factor
        return dwell_action * base_dwelltime + (1 - dwell_action) * max_dwelltime

    def _check_feasibility(
        self, 
        arrival_time: float, 
        departure_time: float
    ) -> bool:
        """
        Check if the proposed times are feasible.
        
        Implements checks for:
        - Headway constraints
        - Overtaking constraints
        - Station capacity constraints
        """
        # Implementation details for feasibility checking
        # This would include your logic for checking train conflicts,
        # headway violations, and other operational constraints
        pass

    def _calculate_reward(
        self,
        action: np.ndarray,
        arrival_time: float,
        departure_time: float
    ) -> float:
        """
        Calculate the reward for the current action.
        
        Considers:
        - Deviation from schedule
        - Action costs
        - Delay penalties
        - Exploration bonus
        """
        dwell_action, runtime_action = action
        
        # Calculate various penalty components
        schedule_deviation = (
            self.config.delay_weight * 
            (abs(arrival_time - self.base_schedule[self.stations[self.current_station_index]][0][1]))
        )
        
        action_penalty = (
            self.config.action_weight * 
            (2 - dwell_action - runtime_action)
        )
        
        exploration_bonus = (
            self.config.exploration_weight * 
            abs(dwell_action - runtime_action)
        )
        
        return -(schedule_deviation + action_penalty) + exploration_bonus

    def _get_observation(self) -> np.ndarray:
        """
        Construct the current observation vector.
        
        Returns:
            observation: Current state observation
        """
        # Implementation for creating observation vector
        pass

    def _save_state(self) -> Dict:
        """Save current state variables."""
        return {
            'current_station_index': self.current_station_index,
            'current_time': self.current_time,
            'times': self.times.copy(),
            'tracked_trains': self.tracked_trains.copy()
        }

    def _restore_state(self, state: Dict):
        """Restore state from saved snapshot."""
        self.current_station_index = state['current_station_index']
        self.current_time = state['current_time']
        self.times = state['times']
        self.tracked_trains = state['tracked_trains']

    def _generate_modified_schedule(self) -> Dict[str, List[Tuple[int, float, float, int]]]:
        """Generate a modified schedule with random delays."""
        # Implementation for creating modified schedules
        pass