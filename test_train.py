#!/usr/bin/env python3
"""
test_train.py — CartPole REINFORCE training harness for dashboard visualization.

Implements CartPole-v1 dynamics from scratch (zero external dependencies beyond
PyTorch, which is already required by the project).  Runs 3 experiments with
different hyperparameters and writes all data in the format the DRL AutoResearch
dashboard expects. The caller decides whether a run is a full baseline/local
eval that should be promoted to registry/dashboard, or a raw-log-only test:

  • logs/experiment_registry.tsv  — ExperimentRegistry (26-column) format, when promoted
  • logs/artifacts/<run_id>/metrics.json  — full training / eval curves
  • logs/runs/<run_id>/metrics.jsonl      — per-step metric stream
  • .drl_autoresearch/state.json          — updated best metric + phase

Usage:
    uv run python test_train.py
    uv run python test_train.py --updates 30   # faster, fewer updates
    uv run python test_train.py --no-registry   # raw logs only
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from drl_autoresearch.core.run_manager import RunManager
from drl_autoresearch.core.state import ProjectState
from drl_autoresearch.logging.registry import ExperimentRegistry

PROJECT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# CartPole-v1 (no gymnasium dependency)
# ---------------------------------------------------------------------------

class CartPole:
    """
    Minimal CartPole-v1 dynamics.
    Compatible with gymnasium CartPole-v1 physics constants.
    """
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    POLE_HALF_LEN = 0.5
    FORCE_MAG = 10.0
    TAU = 0.02          # seconds per step
    MAX_STEPS = 500
    X_LIMIT = 2.4
    THETA_LIMIT = 0.2094  # ~12 degrees in radians

    def reset(self) -> torch.Tensor:
        self._s = torch.rand(4) * 0.1 - 0.05  # uniform [-0.05, 0.05]
        self._n = 0
        return self._s.clone()

    def step(self, action: int):
        x, xd, th, thd = self._s
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        ct, st = math.cos(th.item()), math.sin(th.item())
        totm = self.MASSCART + self.MASSPOLE
        tmp = (force + self.MASSPOLE * self.POLE_HALF_LEN * thd.item() ** 2 * st) / totm
        thacc = (self.GRAVITY * st - ct * tmp) / (
            self.POLE_HALF_LEN * (4.0 / 3.0 - self.MASSPOLE * ct ** 2 / totm)
        )
        xacc = tmp - self.MASSPOLE * self.POLE_HALF_LEN * thacc * ct / totm
        self._s = torch.tensor([
            x.item() + self.TAU * xd.item(),
            xd.item() + self.TAU * xacc,
            th.item() + self.TAU * thd.item(),
            thd.item() + self.TAU * thacc,
        ])
        self._n += 1
        done = (
            abs(self._s[0].item()) > self.X_LIMIT
            or abs(self._s[2].item()) > self.THETA_LIMIT
            or self._n >= self.MAX_STEPS
        )
        reward = 1.0 if not done else 0.0
        return self._s.clone(), reward, done


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(x))


# ---------------------------------------------------------------------------
# REINFORCE update
# ---------------------------------------------------------------------------

def collect_episodes(policy: Policy, env: CartPole, n: int, gamma: float = 0.99):
    """Run n episodes, return (log_probs, returns, mean_reward)."""
    all_lp, all_ret, ep_rewards = [], [], []
    for _ in range(n):
        state = env.reset()
        lp, rew = [], []
        done = False
        while not done:
            dist = policy(state)
            a = dist.sample()
            lp.append(dist.log_prob(a))
            state, r, done = env.step(a.item())
            rew.append(r)
        G, ret = 0.0, []
        for r in reversed(rew):
            G = r + gamma * G
            ret.insert(0, G)
        all_lp.extend(lp)
        all_ret.extend(ret)
        ep_rewards.append(sum(rew))
    return all_lp, all_ret, ep_rewards


def reinforce_step(
    policy: Policy,
    optimizer: optim.Optimizer,
    log_probs,
    returns,
    entropy_coef: float = 0.01,
) -> float:
    ret_t = torch.tensor(returns, dtype=torch.float32)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    lp_t = torch.stack(log_probs)
    loss = -(lp_t * ret_t).mean() - entropy_coef * lp_t.mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return loss.item()


def evaluate(policy: Policy, env: CartPole, n: int = 10) -> tuple[float, float]:
    """Greedy evaluation. Returns (mean, std) reward."""
    rewards = []
    for _ in range(n):
        state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                dist = policy(state)
                a = dist.probs.argmax()
            state, r, done = env.step(a.item())
            ep_r += r
        rewards.append(ep_r)
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# Single-experiment training loop
# ---------------------------------------------------------------------------

def run_experiment(
    experiment: dict,
    project_dir: Path,
    run_manager: RunManager,
    registry: ExperimentRegistry,
    state: ProjectState,
    total_updates: int = 50,
    publish_to_registry: bool = True,
    registry_decision_reason: str = "agent marked this as a full local eval/baseline result",
) -> None:
    experiment = {
        **experiment,
        "publish_to_registry": publish_to_registry,
        "registry_decision_reason": registry_decision_reason,
    }
    params = experiment["params"]
    lr = params["learning_rate"]
    hidden = params.get("hidden_size", 64)
    eps_per_update = params.get("batch_eps", 8)
    gamma = params.get("gamma", 0.99)

    policy = Policy(hidden=hidden)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    env = CartPole()

    ctx = run_manager.start_run(experiment)
    t0 = time.time()

    print(
        f"\n[~] {ctx.run_id}  "
        f"lr={lr}  hidden={hidden}  eps/update={eps_per_update}"
    )
    print(f"    hypothesis: {experiment.get('hypothesis', '')}")

    steps_log, reward_log, loss_log = [], [], []
    eval_steps_log, eval_reward_log = [], []
    global_step = 0
    best_eval = 0.0

    for upd in range(total_updates):
        lp, ret, ep_rew = collect_episodes(policy, env, n=eps_per_update, gamma=gamma)
        loss_val = reinforce_step(policy, optimizer, lp, ret)

        global_step += eps_per_update
        mean_r = sum(ep_rew) / len(ep_rew)

        steps_log.append(global_step)
        reward_log.append(round(mean_r, 2))
        loss_log.append(round(loss_val, 4))

        run_manager.log_metric(ctx, step=global_step, metrics={
            "train_reward_mean": mean_r,
            "loss": loss_val,
        })

        # Eval every 10 updates and at the end
        if (upd + 1) % 10 == 0 or upd == total_updates - 1:
            emean, estd = evaluate(policy, env, n=10)
            eval_steps_log.append(global_step)
            eval_reward_log.append(round(emean, 2))
            best_eval = max(best_eval, emean)
            print(
                f"  upd {upd+1:3d}/{total_updates}  "
                f"train={mean_r:6.1f}  eval={emean:6.1f}±{estd:.1f}  "
                f"loss={loss_val:+.4f}"
            )

    wall = time.time() - t0

    # ---- Write full curves artifact (dashboard reads this) ----
    artifact_dir = project_dir / "logs" / "artifacts" / ctx.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(
        json.dumps({
            "steps": steps_log,
            "rewards": reward_log,
            "losses": loss_log,
            "eval_steps": eval_steps_log,
            "eval_rewards": eval_reward_log,
        }, indent=2),
        encoding="utf-8",
    )

    # ---- Finish run → RunRecord ----
    final_eval_mean = eval_reward_log[-1] if eval_reward_log else None
    final_eval_std = 5.0
    train_tail = reward_log[-5:] if len(reward_log) >= 5 else reward_log
    train_mean = sum(train_tail) / len(train_tail)
    train_std = (sum((r - train_mean) ** 2 for r in train_tail) / len(train_tail)) ** 0.5

    result = {
        "eval_reward_mean": final_eval_mean,
        "eval_reward_std": final_eval_std,
        "train_reward_mean": train_mean,
        "train_reward_std": train_std,
        "wall_clock_seconds": wall,
        "algorithm": "REINFORCE",
        "environment": "CartPole-v1",
        "change_summary": experiment.get("change_summary", ""),
        "agent": "test_train.py",
        "branch": "master",
        "status": "completed",
        "publish_to_registry": publish_to_registry,
        "registry_decision_reason": registry_decision_reason,
        "notes": f"best_eval={best_eval:.1f}",
    }

    run_record = run_manager.finish_run(ctx, result)

    # keep_decision based on whether eval improved vs previous best
    if state.best_metric_value is None or (final_eval_mean or 0) >= state.best_metric_value:
        run_record.keep_decision = "keep"
        decision_label = "KEEP"
    else:
        run_record.keep_decision = "discard"
        decision_label = "discard"

    publish_to_registry = run_manager.should_publish_to_registry(ctx, result)
    if publish_to_registry:
        registry.add_run(run_record)
    else:
        print(f"  [raw-log-only] {ctx.run_id}; registry/dashboard skipped: {registry_decision_reason}")
        return

    # Update state
    state.total_runs += 1
    if run_record.keep_decision == "keep":
        state.kept_runs += 1
        if final_eval_mean is not None:
            improved = state.update_best(ctx.run_id, final_eval_mean, "eval_reward_mean")
            if improved:
                print(f"  [✓] New best: eval_reward_mean={final_eval_mean:.1f}")
    else:
        state.discarded_runs += 1

    state.save()

    print(
        f"  [{decision_label}] {ctx.run_id}  "
        f"eval={final_eval_mean:.1f}  wall={wall:.0f}s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "hypothesis": "Baseline — default REINFORCE on CartPole",
        "change_summary": "Default hyperparameters; establishes performance floor.",
        "params": {
            "learning_rate": 1e-3,
            "hidden_size": 64,
            "batch_eps": 8,
            "gamma": 0.99,
        },
    },
    {
        "hypothesis": "Higher LR — faster convergence at the cost of stability",
        "change_summary": "3× higher learning rate; tests if faster updates help.",
        "params": {
            "learning_rate": 3e-3,
            "hidden_size": 64,
            "batch_eps": 8,
            "gamma": 0.99,
        },
    },
    {
        "hypothesis": "Wider network — more capacity for value estimation",
        "change_summary": "hidden_size 64→128; tests whether capacity is the bottleneck.",
        "params": {
            "learning_rate": 1e-3,
            "hidden_size": 128,
            "batch_eps": 8,
            "gamma": 0.99,
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test training harness for DRL AutoResearch dashboard.")
    parser.add_argument("--updates", type=int, default=50, help="Gradient updates per experiment (default: 50)")
    parser.add_argument("--project-dir", default=".", metavar="DIR")
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Keep results in raw logs only; do not publish to registry/dashboard.",
    )
    parser.add_argument(
        "--registry-reason",
        default="agent marked this as a full local eval/baseline result",
        help="Short agent decision reason for publishing or skipping registry/dashboard.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    total_updates = args.updates

    print("=" * 60)
    print("  DRL AutoResearch — Test Training Harness")
    print(f"  Project : {project_dir}")
    print(f"  Updates : {total_updates} per experiment")
    print(f"  Runs    : {len(EXPERIMENTS)}")
    print(f"  Registry: {'skip' if args.no_registry else 'publish'}")
    print("=" * 60)

    run_manager = RunManager(project_dir=project_dir)
    registry = ExperimentRegistry(project_dir=project_dir)
    registry.initialize()
    state = ProjectState.load(project_dir)

    t_total = time.time()
    for exp in EXPERIMENTS:
        run_experiment(
            experiment=exp,
            project_dir=project_dir,
            run_manager=run_manager,
            registry=registry,
            state=state,
            total_updates=total_updates,
            publish_to_registry=not args.no_registry,
            registry_decision_reason=args.registry_reason,
        )

    print(f"\n{'=' * 60}")
    print(f"  All {len(EXPERIMENTS)} experiments complete in {time.time()-t_total:.0f}s")
    print(f"  Best eval_reward_mean : {state.best_metric_value}")
    print(f"  Best run              : {state.best_run_id}")
    print(f"  Phase                 : {state.current_phase}")
    print(f"  Dashboard             : http://localhost:8765")
    print("=" * 60)


if __name__ == "__main__":
    main()
