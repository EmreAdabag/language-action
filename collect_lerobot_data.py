import sys
import json as _json
from pathlib import Path
import torch
from env import PickAndPlaceEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse
import time
import numpy as np


def collect_lerobot_dataset(
    episodes_per_task: int,
    root: str | Path,
    tasks: list,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Action is 2D discrete movement (dx, dy)
    features = {
        "action": {
            "dtype": "int64",
            "shape": (2,),
            "names": ["dx", "dy"],
        },
        "grid": {
            "dtype": "uint8",
            "shape": (64,),
            "names": None,
        }
    }

    root = Path(root)
    ds = LeRobotDataset.create(
        repo_id='null',
        fps=1,
        features=features,
        root=root
    )
    
    # Store sprite_names once in the dataset metadata
    env_temp = PickAndPlaceEnv(device=device, seed=seed)
    sprite_names = env_temp.sprite_names
    sidecar_path = Path(root) / "sprite_names.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        _json.dump(sprite_names, fh, indent=2)

    max_steps = 50
    episode_task_records = []

    for i, (obj_idx, goal_idx) in enumerate(tasks):
        print(f"\n\nstarting task {i}\n")
        task_s = time.time()

        env = PickAndPlaceEnv(device=device, seed=seed + i)
        
        task_frames = 0
        task_successes = 0
        episode_lengths = []

        for ep in range(episodes_per_task):
            env.reset(obj_idx=obj_idx, goal_idx=goal_idx)
            
            task_str = env.command()
            
            steps = 0
            while steps < max_steps:
                # Simple expert: move toward goal
                # object_pos and goal_pos are [row, col]
                obj_r, obj_c = env.object_pos[0].cpu()
                goal_r, goal_c = env.goal_pos[0].cpu()
                
                # dx = change in column (X axis), dy = change in row (Y axis)
                dx = torch.clamp(goal_c - obj_c, -1, 1)
                dy = torch.clamp(goal_r - obj_r, -1, 1)
                action = torch.tensor([[dx, dy]], device=device, dtype=torch.long)
                
                env.step(action)

                frame = {
                    "action": action[0].cpu().numpy(),
                    "task": task_str,
                    "grid": env.grid[0].flatten().cpu().numpy().astype(np.uint8),
                }
                ds.add_frame(frame)
                task_frames += 1
                
                if env.success[0].item():
                    task_successes += 1
                    break
                steps += 1

            episode_lengths.append(steps + 1)
            ds.save_episode()
            episode_task_records.append({
                "task": task_str,
            })

        task_e = time.time()
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"Task {i} completed in {task_e - task_s:.2f} seconds")
        print(f"  Episodes: {episodes_per_task}")
        print(f"  Total frames: {task_frames}")
        print(f"  Avg frames/episode: {task_frames / episodes_per_task:.1f}")
        print(f"  Avg episode length: {avg_length:.1f} steps")
        print(f"  Success rate: {task_successes}/{episodes_per_task} ({100*task_successes/episodes_per_task:.1f}%)")
    
    ds.finalize()
    
    # Save episode-level task strings next to the dataset
    sidecar_path = Path(root) / "episode_tasks.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        _json.dump(episode_task_records, fh, indent=2)
    print(f"LeRobot dataset recorded: root='{root}', episodes={episodes_per_task*len(tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pick+place demos for a fixed object over a goal range")
    parser.add_argument("--object_id_s", type=int, required=True)
    parser.add_argument("--object_id_e", type=int, required=True)
    parser.add_argument("--goal_id_s", type=int, required=True)
    parser.add_argument("--goal_id_e", type=int, required=True)
    parser.add_argument("--root", type=str, default="datasets/tmp")
    parser.add_argument("--episodes_per_task", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    omit = set([(28,132), (29,133), (30,134), (31,134)])
    tasks = list(omit)
    # tasks = [(o, g) for g in range(args.goal_id_s, args.goal_id_e) for o in range(args.object_id_s, args.object_id_e) if g != o and (o, g) not in omit]
    print(f"Collecting {len(tasks)} tasks")
    
    collect_lerobot_dataset(
        episodes_per_task=args.episodes_per_task,
        root=args.root,
        tasks=tasks,
        seed=args.seed,
    )

