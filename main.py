import wandb
from craftground import craftground
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from action_wrapper import ActionWrapper, Action
from avoid_damage import AvoidDamageWrapper
from fast_reset import FastResetWrapper
from time_limit_wrapper import TimeLimitWrapper
from vision_wrapper import VisionWrapper


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log training information to wandb
        wandb.log(
            {
                "total_timesteps": self.num_timesteps,
                "episode_reward": self.locals["episode_reward"],
            }
        )
        return True


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="escape-husk",
    )
    env = craftground.make(
        # env_path="../minecraft_env",
        port=8023,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            # player looks at south (positive Z) when spawn
        ],
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=114,
        visibleSizeY=64,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=True,
        alwaysNight=False,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=[],
        isHudHidden=False,
        render_action=True,
        render_distance=2,
        simulation_distance=5,
    )
    env = FastResetWrapper(
        TimeLimitWrapper(
            ActionWrapper(
                AvoidDamageWrapper(VisionWrapper(env, x_dim=114, y_dim=64)),
                enabled_actions=[Action.FORWARD, Action.BACKWARD],
            ),
            max_timesteps=400,
        )
    )

    model = A2C("MlpPolicy", env, verbose=1, device="mps")
    wandb_callback = WandbCallback()
    model.learn(total_timesteps=10000, callback=wandb_callback)
    model.save("a2c_craftground")

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     # vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()


if __name__ == "__main__":
    main()
