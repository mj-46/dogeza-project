"""isaaclab_学習用"""

"""後半（土下座 → 跪座）の PPO の設定。前半の設定（first_half/agent_cfg.py）を引き継ぎ、違うところだけを書く。"""

from isaaclab.utils import configclass

from dogeza_rl.first_half.agent_cfg import DogezaPPORunnerCfg


@configclass
class DogezaUpPPORunnerCfg(DogezaPPORunnerCfg):
    max_iterations = 3000  # 更新の回数（起き上がるだけなので前半より短め）
    experiment_name = "dogeza_g1_up"  # 保存先: logs/rsl_rl/dogeza_g1_up/
