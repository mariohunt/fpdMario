from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

ACTIONS_8 = [
    (-1,  0),
    ( 1,  0),
    ( 0, -1),
    ( 0,  1),
    (-1, -1),
    (-1,  1),
    ( 1, -1),
    ( 1,  1),
]

@dataclass
class GridWorld:
    n_rows: int
    n_cols: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    walls: List[Tuple[int, int]]
    portal_in: Tuple[int, int]
    portal_out: Tuple[int, int]
    step_reward: float = -1.0
    wall_reward: float = -2.0
    goal_reward: float = 50.0
    portal_reward: float = -0.5

    def reset(self) -> Tuple[int, int]:
        return self.start

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, state: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
        dr, dc = ACTIONS_8[action_idx]
        r, c = state
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc):
            return state, self.wall_reward, False
        ns = (nr, nc)
        if ns in set(self.walls):
            return state, self.wall_reward, False
        if ns == self.portal_in:
            return self.portal_out, self.portal_reward, False
        if ns == self.goal:
            return ns, self.goal_reward, True
        return ns, self.step_reward, False

    def valid_actions(self, state: Tuple[int, int]) -> List[int]:
        val = []
        w = set(self.walls)
        r, c = state
        for i, (dr, dc) in enumerate(ACTIONS_8):
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc):
                continue
            if (nr, nc) in w:
                continue
            val.append(i)
        return val

def epsilon_greedy_with_momentum(Q: np.ndarray, s_idx: int, valid: List[int], epsilon: float,
                                prev_action: Optional[int], momentum_p: float, rng: np.random.Generator) -> int:
    if prev_action is not None and prev_action in valid and rng.random() < momentum_p:
        return prev_action
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    qvals = Q[s_idx, valid]
    return valid[int(np.argmax(qvals))]

def train_q_learning(env: GridWorld,
                     episodes: int = 1500,
                     alpha: float = 0.25,
                     gamma: float = 0.95,
                     eps_start: float = 1.0,
                     eps_min: float = 0.05,
                     eps_decay: float = 0.995,
                     momentum_p: float = 0.65,
                     max_steps: int = 250,
                     seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_states = env.n_rows * env.n_cols
    n_actions = len(ACTIONS_8)

    def s_to_idx(s: Tuple[int, int]) -> int:
        return s[0] * env.n_cols + s[1]

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    rewards = np.zeros(episodes, dtype=np.float32)
    steps = np.zeros(episodes, dtype=np.int32)

    eps = eps_start
    for ep in range(episodes):
        s = env.reset()
        prev_a = None
        total = 0.0
        for t in range(max_steps):
            s_idx = s_to_idx(s)
            valid = env.valid_actions(s)
            a = epsilon_greedy_with_momentum(Q, s_idx, valid, eps, prev_a, momentum_p, rng)
            ns, r, done = env.step(s, a)
            ns_idx = s_to_idx(ns)
            best_next = np.max(Q[ns_idx])
            td_target = r + (0.0 if done else gamma * best_next)
            Q[s_idx, a] += alpha * (td_target - Q[s_idx, a])
            total += r
            s = ns
            prev_a = a
            if done:
                steps[ep] = t + 1
                break
        else:
            steps[ep] = max_steps
        rewards[ep] = total
        eps = max(eps_min, eps * eps_decay)

    return {"Q": Q, "episode_rewards": rewards, "episode_steps": steps}

def greedy_policy_path(env: GridWorld, Q: np.ndarray, max_steps: int = 200) -> List[Tuple[int, int]]:
    def s_to_idx(s: Tuple[int, int]) -> int:
        return s[0] * env.n_cols + s[1]

    s = env.reset()
    path = [s]
    for _ in range(max_steps):
        if s == env.goal:
            break
        s_idx = s_to_idx(s)
        valid = env.valid_actions(s)
        a = valid[int(np.argmax(Q[s_idx, valid]))]
        s, _, done = env.step(s, a)
        path.append(s)
        if done:
            break
    return path

def default_env() -> GridWorld:
    walls = [(1,1),(1,2),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(6,4),(6,5),(2,6),(3,6)]
    return GridWorld(
        n_rows=8,
        n_cols=8,
        start=(0,0),
        goal=(7,7),
        walls=walls,
        portal_in=(3,1),
        portal_out=(6,6),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    args = ap.parse_args()
    env = default_env()
    out = train_q_learning(env, episodes=args.episodes)
    path = greedy_policy_path(env, out["Q"])
    print(float(np.mean(out["episode_rewards"][-100:])))
    print(float(np.mean(out["episode_steps"][-100:])))
    print(path[:15])

if __name__ == "__main__":
    main()
