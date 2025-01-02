from Rail_environment_RL.environment_RL import TrainNetworkEnv, TrainConfig

# Define network
stations = ['Station1', 'Station2', 'Station3']
runtimes = {
    'Station1': 0.1,
    'Station2': 0.15,
    'Station3': 0.12
}
dwelltimes = {
    'Station1': 0.02,
    'Station2': 0.03,
    'Station3': 0.02
}

# Create base schedule
base_schedule = {
    'Station1': [(1, 0.0, 0.02, 1)],  # (train_id, arrival, departure, direction)
    'Station2': [(1, 0.12, 0.15, 1)],
    'Station3': [(1, 0.27, 0.29, 1)]
}

# Create custom config
config = TrainConfig(
    beta_waiting=0.8,
    beta_running=0.8,
    headway_time=0.002
)

# Initialize environment
env = TrainNetworkEnv(
    stations=stations,
    runtimes=runtimes,
    dwelltimes=dwelltimes,
    base_schedule=base_schedule,
    config=config
)