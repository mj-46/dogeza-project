import gymnasium

gymnasium.register(
    id="Dogeza-G1",
    entry_point="dogeza_rl.env:DogezaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dogeza_rl.env_cfg:DogezaEnvCfg",
        "rsl_rl_cfg_entry_point": "dogeza_rl.agent_cfg:DogezaPPORunnerCfg",
    },
)
