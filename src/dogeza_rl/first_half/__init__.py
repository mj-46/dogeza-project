"""isaaclab_学習用"""

"""前半（立つ → 土下座）のタスクの登録。"""

import gymnasium as gym

gym.register(
    id="dogeza-first-half",
    entry_point="dogeza_rl.first_half.env:DogezaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dogeza_rl.first_half.env_cfg:DogezaEnvCfg",
        "rsl_rl_cfg_entry_point": "dogeza_rl.first_half.agent_cfg:DogezaPPORunnerCfg",
    },
)
