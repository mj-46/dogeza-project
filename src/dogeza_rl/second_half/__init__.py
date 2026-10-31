"""isaaclab_学習用"""

"""後半（土下座 → 跪座）のタスクの登録。"""

import gymnasium as gym

gym.register(
    id="dogeza-second-half",
    entry_point="dogeza_rl.second_half.env:DogezaUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dogeza_rl.second_half.env_cfg:DogezaUpEnvCfg",
        "rsl_rl_cfg_entry_point": "dogeza_rl.second_half.agent_cfg:DogezaUpPPORunnerCfg",
    },
)
