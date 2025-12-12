from pathlib import Path
import torch


class PickAndPlaceEnv:
    """
    Text-based 8x8 grid environment.
    - Grid is filled with text tokens (sprite names)
    - Action: (dx, dy) discrete movement of object
    - Command: move object token to goal token position
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        batch_size: int = 1,
        objects_dir: str | Path = "objects",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = torch.Generator(device=device).manual_seed(int(seed))
        self.B = int(batch_size)

        # Grid
        self.grid_size = 8

        # Load sprite names
        self.objects_dir = Path(objects_dir)
        self.sprite_names = self._load_sprite_names(self.objects_dir)
        assert len(self.sprite_names) > 0, f"No PNG sprites found in '{self.objects_dir}'."

        # State
        self.grid = None              # [B, 8, 8] long (index into sprite_names)
        self.object_pos = None        # [B, 2] long (row, col)
        self.goal_pos = None          # [B, 2] long (row, col)
        self.object_id = None         # [B] long (index into sprite_names)
        self.goal_id = None           # [B] long (index into sprite_names)
        self.success = None           # [B] bool

        self.reset()

    def reset(self, obj_idx=None, goal_idx=None):
        B = self.B
        N = len(self.sprite_names)

        # Random object and goal positions
        self.object_pos = torch.randint(0, self.grid_size, (B, 2), device=self.device, generator=self.rng)
        self.goal_pos = torch.randint(0, self.grid_size, (B, 2), device=self.device, generator=self.rng)

        # Select object and goal sprites
        if obj_idx is not None:
            self.object_id = torch.full((B,), obj_idx, device=self.device, dtype=torch.long)
        else:
            self.object_id = torch.randint(0, N, (B,), device=self.device, generator=self.rng)

        if goal_idx is not None:
            self.goal_id = torch.full((B,), goal_idx, device=self.device, dtype=torch.long)
        else:
            self.goal_id = torch.randint(0, N, (B,), device=self.device, generator=self.rng)

        # fill grid with the number 144
        self.grid = torch.full((B, self.grid_size, self.grid_size), 144, device=self.device, dtype=torch.long)
        # self.grid = torch.randint(0, N, (B, self.grid_size, self.grid_size), device=self.device, generator=self.rng)
        # mask = (self.grid == self.object_id.view(B, 1, 1)) | (self.grid == self.goal_id.view(B, 1, 1))
        # while mask.any():
        #     self.grid[mask] = torch.randint(0, N, (mask.sum().item(),), device=self.device, generator=self.rng)
        #     mask = (self.grid == self.object_id.view(B, 1, 1)) | (self.grid == self.goal_id.view(B, 1, 1))

        # Place object sprite at object position
        batch_idx = torch.arange(B, device=self.device)
        self.grid[batch_idx, self.object_pos[:, 0], self.object_pos[:, 1]] = self.object_id
        self.grid[batch_idx, self.goal_pos[:, 0], self.goal_pos[:, 1]] = self.goal_id

        self.success = torch.zeros(B, dtype=torch.bool, device=self.device)
        return None

    def step(self, action: torch.Tensor):
        # action: [B, 2] containing (dx, dy) in {-1, 0, 1}
        # dx = change in column (X axis, horizontal)
        # dy = change in row (Y axis, vertical)
        a = action.to(device=self.device, dtype=torch.long)
        dx = a[:, 0].clamp(-1, 1)
        dy = a[:, 1].clamp(-1, 1)

        # Update object position with discrete single unit moves
        new_pos = self.object_pos.clone()
        new_pos[:, 0] = (self.object_pos[:, 0] + dy).clamp(0, self.grid_size - 1)
        new_pos[:, 1] = (self.object_pos[:, 1] + dx).clamp(0, self.grid_size - 1)

        # Update grid: clear old position, set new position
        batch_idx = torch.arange(self.B, device=self.device)
        self.grid[batch_idx, self.object_pos[:, 0], self.object_pos[:, 1]] = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]]
        self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]] = self.object_id

        self.object_pos = new_pos

        # Check success: object at goal position
        self.success = (self.object_pos == self.goal_pos).all(dim=1)

    def pretty_print(self, batch_idx=0):
        """Print the grid for a specific batch element."""
        b = batch_idx
        grid_cpu = self.grid[b].cpu()
        obj_r, obj_c = self.object_pos[b].cpu()
        goal_r, goal_c = self.goal_pos[b].cpu()

        lines = []
        lines.append("+" + "-" * (self.grid_size * 12 + 1) + "+")
        for r in range(self.grid_size):
            row = "|"
            for c in range(self.grid_size):
                token_id = grid_cpu[r, c].item()
                token = self.sprite_names[token_id].ljust(10)
                
                # Mark object and goal positions
                if r == obj_r and c == obj_c:
                    row += f"*{token}*"
                elif r == goal_r and c == goal_c:
                    row += f"[{token}]"
                else:
                    row += f" {token} "
            row += "|"
            lines.append(row)
        lines.append("+" + "-" * (self.grid_size * 12 + 1) + "+")
        lines.append(f"Command: {self.command()}")
        return "\n".join(lines)

    def command(self) -> str:
        # Assumes batch size 1 usage for commands
        obj_name = self.sprite_names[self.object_id[0].item()]
        goal_name = self.sprite_names[self.goal_id[0].item()]
        return f"place the {obj_name} on the {goal_name}"

    def _load_sprite_names(self, directory: Path):
        names = []
        for p in sorted(directory.glob("*.png")):
            names.append(" " + p.stem)
        return names

    def get_env_state(self):
        state = {
            "grid": self.grid.cpu(),
            "object_pos": self.object_pos.cpu(),
            "goal_pos": self.goal_pos.cpu(),
            "object_id": self.object_id.cpu(),
            "goal_id": self.goal_id.cpu(),
            "success": self.success.cpu(),
        }
        return state

    def set_env_state(self, state):
        self.grid = state["grid"].to(device=self.device, dtype=torch.long)
        self.object_pos = state["object_pos"].to(device=self.device, dtype=torch.long)
        self.goal_pos = state["goal_pos"].to(device=self.device, dtype=torch.long)
        self.object_id = state["object_id"].to(device=self.device, dtype=torch.long)
        self.goal_id = state["goal_id"].to(device=self.device, dtype=torch.long)
        self.success = state["success"].to(device=self.device, dtype=torch.bool)

