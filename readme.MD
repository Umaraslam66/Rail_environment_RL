Environment Details
State Space
The observation space includes:

Current station (one-hot encoded)
Current time
Nearby train positions
Schedule adherence information
Constraint violation flags

Action Space
The action space is continuous and includes:

Running time adjustment factor
Dwell time adjustment factor

Reward Function
The reward function considers:

Schedule deviation
Energy efficiency
Passenger comfort
Operational constraints

Configuration
Use TrainConfig to customize environment parameters:

config = TrainConfig(
    beta_waiting=0.7,  # Recovery rate for waiting time
    beta_running=0.7,  # Recovery rate for running time
    headway_time=0.0013889,  # Minimum headway time
    max_runtime_factor=2.5  # Maximum runtime extension
)