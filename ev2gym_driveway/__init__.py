from gymnasium.envs.registration import register

register(
    id='EV2Gym-Driveway-v1',
    entry_point='ev2gym_driveway.models.ev2gym_driveway_env:EV2GymDriveway',
    kwargs={'config_file': 'ev2gym_driveway/example_config_files/V2GProfitMax.yaml'}
)