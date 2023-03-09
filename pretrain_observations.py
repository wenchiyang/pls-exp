import os
from pls.workflows.execute_workflow import pretrain_observation_stars, pretrain_observation_cr
from dask.distributed import Client
from pls.observation_nets.observation_nets import Observation_Net_Stars, Observation_Net_Carracing
import gym
import pacman_gym
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from pacman_gym.envs.goal_finding import sample_layout
import csv
from pls.dpl_policies.pacman.util import get_ground_wall, get_agent_coord
from pls.dpl_policies.carracing.util import get_ground_truth_of_grass
from pls.workflows.evaluate import load_model_and_env
import json
from matplotlib import pyplot as plt
import torch as th

def generate_random_images_cr(csv_path, folder, n_images=10):
    def load_policy_cr(folder, model_at_step):
        path = os.path.join(folder, "config.json")
        with open(path) as json_data_file:
            config = json.load(json_data_file)
        model, env = load_model_and_env(folder, config, model_at_step)
        return model, env
    # Use a pretrained (600k steps) agent to generate states
    policy_folder = os.path.join(os.path.dirname(__file__), "experiments/carracing/map0/PPO/seed1")
    model, env = load_policy_cr(policy_folder, model_at_step=600000)

    deterministic=False
    render=True
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "grass(in_front)","grass(on_the_left)", "grass(on_the_right)"])

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    observations = env.reset()
    current_lengths = 0
    n = 0
    while n < n_images:
        actions = model.predict(observations, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions[0])

        if render:
            for e in env.envs:
                e.env.render()

        if current_lengths % 50 == 0:
            img = env.envs[0].render(mode="state_pixels")
            gray_img = env.envs[0].render(mode="gray")
            gray_img = th.tensor(gray_img).unsqueeze(dim=0).unsqueeze(dim=1)

            path = os.path.join(folder, f"img{n:06}.png")

            plt.imsave(path, img)
            ground_truth_grass = get_ground_truth_of_grass(input=gray_img)
            row = [f"img{n:06}.png"] + ground_truth_grass.flatten().tolist()
            writer.writerow(row)
            f_csv.flush()
            n += 1

        current_lengths += 1

def generate_pacman(num_imgs, ghost_distance, img_folder, map_name="small"):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    csv_file = os.path.join(img_folder, "labels.csv")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1
    f_csv = open(csv_file, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)", "agent_r", "agent_c"])
    config = {
        "env_type": "GoalFinding-v0",
        "env_features":{
            "layout": map_name,
            "reward_goal": 10,
            "reward_crash": 0,
            "reward_food": 0,
            "reward_time": -0.1,
            "render": False,
            "max_steps": 2000,
            "num_maps": 0,
            "seed": 567,
            'render_mode': "gray",
            "height": 482,
            "width": 482,
            "downsampling_size": 1,
            "background": "bg_small.jpg"
        },
        "shield_params": {
            "ghost_distance": ghost_distance
        }
    }
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)
    env.env.gameDisplay = env.env.display
    env.env.rules.quiet = False

    num_ghosts = 30 if map_name == "small" else env.env.num_agents
    num_food = 30 if map_name == "small" else env.env.num_food
    for n in range(num_imgs):
        layout = sample_layout(
            env.layout.width,
            env.layout.height,
            num_ghosts,
            num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges,
            check_valid=False
        )
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
            env.env.background
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("rgb")
        path = os.path.join(img_folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.game.compose_img("tinygrid")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_ghost = get_ground_wall(tinyGrid, PACMAN_COLOR, GHOST_COLOR, ghost_distance)
        agent_r, agent_c = get_agent_coord(tinyGrid, PACMAN_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
        writer.writerow(row)
        f_csv.flush()
        if (n+1) % 10 == 0:
            print(f'Produce: {n+1}/{num_imgs} [({float(n+1)/num_imgs*100:.0f}%)]')

    f_csv.close()

def generate_cr(num_imgs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "../pls/data/carracing")

    csv_file = os.path.join(img_folder, "labels.csv")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    generate_random_images_cr(csv_file, img_folder, num_imgs)

def generate_stars(num_imgs, img_folder):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    csv_file = os.path.join(img_folder, "labels.csv")
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1
    f_csv = open(csv_file, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)", "agent_r", "agent_c"])
    config = {
        "env_type": "GoalFinding-v0",
        "env_features":{
            "layout": "small",
            "reward_goal": 10,
            "reward_crash": 0,
            "reward_food": 0,
            "reward_time": -0.1,
            "render": False,
            "max_steps": 2000,
            "num_maps": 0,
            "seed": 567,
            'render_mode': "gray",
            "height": 482,
            "width": 482,
            "downsampling_size": 1,
            "background": "bg_small.jpg"
        },
        "shield_params": {
            "ghost_distance": 1
        }
    }
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)
    env.env.gameDisplay = env.env.display
    env.env.rules.quiet = False

    for n in range(num_imgs):
        layout = sample_layout(
            env.layout.width,
            env.layout.height,
            30, #env.env.num_agents,
            30, #env.env.num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges,
            check_valid=False
        )
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
            env.env.background
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("rgb")
        path = os.path.join(img_folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.game.compose_img("tinygrid")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_ghost = get_ground_wall(tinyGrid, PACMAN_COLOR, GHOST_COLOR, config["shield_params"]["ghost_distance"])
        agent_r, agent_c = get_agent_coord(tinyGrid, PACMAN_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
        writer.writerow(row)
        f_csv.flush()
        if (n+1) % 10 == 0:
            print(f'Produce: {n+1}/{num_imgs} [({float(n+1)/num_imgs*100:.0f}%)]')

    f_csv.close()

def pre_train_stars(n_train, net_class, epochs, img_folder, model_folder, downsampling_size=None):
    csv_file = os.path.join(img_folder, "labels.csv")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    image_dim = 482
    pretrain_observation_stars(csv_file=csv_file, img_folder=img_folder, model_folder=model_folder,
                               image_dim=image_dim, downsampling_size=downsampling_size,
                               n_train=n_train, epochs=epochs, net_class=net_class)
def pre_train_cr(n_train, net_class, epochs, downsampling_size=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "../pls/data/carracing")

    csv_file = os.path.join(img_folder, "labels.csv")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    model_folder = os.path.join(dir_path, "../experiments_safety/carracing/map1/data/")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    image_dim = 48
    pretrain_observation_cr(csv_file=csv_file, img_folder=img_folder, model_folder=model_folder,
                                 image_dim=image_dim, downsampling_size=downsampling_size,
                                 n_train=n_train, epochs=epochs, net_class=net_class)


def main_stars():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "data/stars")
    model_folder = os.path.join(dir_path, "experiments/stars/small0/data/")
    # generate_stars(num_imgs=3000, img_folder=img_folder)
    pre_train_stars(n_train=500, net_class=Observation_Net_Stars, downsampling_size=8, img_folder=img_folder, model_folder=model_folder, epochs=10)

def main_pacman():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "data/pacman")
    model_folder = os.path.join(dir_path, "experiments/pacman/small3/data/")
    # generate_pacman(num_imgs=2000, ghost_distance=1, img_folder=img_folder, map_name="small4")
    pre_train_stars(n_train=1000, net_class=Observation_Net_Stars, downsampling_size=8, img_folder=img_folder, model_folder=model_folder, epochs=5)

def carracing():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "data/carracing")
    model_folder = os.path.join(dir_path, "experiments/carracing/small0/data/")
    generate_cr(num_imgs=500)
    pre_train_stars(n_train=500, net_class=Observation_Net_Carracing, downsampling_size=1, img_folder=img_folder, model_folder=model_folder, epochs=10)


if __name__ == "__main__":
    carracing()


