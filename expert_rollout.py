import argparse
import torch

from env import PickAndPlaceEnv


class ExpertController:
    """Simple greedy controller for discrete grid navigation."""

    def act(self, env: PickAndPlaceEnv) -> torch.Tensor:
        # Move toward goal one step at a time
        # object_pos and goal_pos are [row, col]
        obj_r, obj_c = env.object_pos[0]
        goal_r, goal_c = env.goal_pos[0]
        
        # dx = change in column (X axis), dy = change in row (Y axis)
        dx = torch.clamp(goal_c - obj_c, -1, 1)
        dy = torch.clamp(goal_r - obj_r, -1, 1)
        
        return torch.tensor([[dx, dy]], device=env.device, dtype=torch.long)


def main():
    for seed in range(10):
        parser = argparse.ArgumentParser(description="Roll out expert policy and print grid")
        parser.add_argument("--max_steps", type=int, default=50, help="Max steps")
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = PickAndPlaceEnv(batch_size=1, device=device, seed=seed)
        ctrl = ExpertController()

        env.reset()
        print("\nInitial state:")
        print(env.pretty_print())
        print()
        
        steps = 0
        while steps < args.max_steps:
            act = ctrl.act(env)
            env.step(act)
            
            print(f"\nStep {steps + 1}, action: {act[0].tolist()}")
            print(env.pretty_print())
            print()
            
            if bool(env.success.all()):
                print(f"Success in {steps + 1} steps!")
                break
            steps += 1

        if not env.success[0]:
            print(f"Failed to complete in {args.max_steps} steps")


if __name__ == "__main__":
    main()
