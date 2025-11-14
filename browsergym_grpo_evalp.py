#!/usr/bin/env python3
"""
GRPO training on BrowserGym using eval_protocol's OpenEnv rollout integration.

User responsibilities:
- Provide env_factory(), prompt_builder(observation, step, history), action_parser(response_text)

Everything else (dataset→EvaluationRows, agent loop, concurrency, rewards, tracing)
is handled by eval_protocol.
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, List, Tuple, Optional

import numpy as np
import logging
from datasets import Dataset
from transformers import AutoProcessor
from trl import GRPOConfig
from peft import LoraConfig
import trl as _trl

# Add OpenEnv and eval_protocol to path if needed
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent / "OpenEnv" / "src"))
# sys.path.insert(0, str(Path(__file__).parent.parent / "python-sdk"))
# sys.path.insert(0, "/home/shrey/trl-eval-protocol-gsm8k-grpo")

from eval_protocol.pytest.integrations.openenv_trl import create_openenv_rollout_func
from fireworks_openenv_trainer import create_openenv_trainer_class
from trl import GRPOTrainer as _BaseTRLTrainer

# Configure logging to surface HotLoad debug/info messages
_hotload_debug = os.getenv("HOTLOAD_DEBUG", "0").lower() in {"1", "true", "yes"}
logging.basicConfig(
    level=logging.DEBUG if _hotload_debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.info("[Init] Logging configured (HOTLOAD_DEBUG=%s)", _hotload_debug)


# ---------------------------------------------------------------------------
# Minimal inline config (sample values for PR)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-8B"
INFER_MODEL_ID = "fireworks_ai/accounts/fireworks/models/qwen3-8b#accounts/pyroworks/deployments/h6mxm330"
MAX_STEPS = 2
MAX_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
LEARNING_RATE = 3e-5
PER_DEVICE_BATCH_SIZE = 6
NUM_GENERATIONS = 2
NUM_EPOCHS = 1
DATASET_SIZE = 3
OUTPUT_DIR = "outputs/openenv-browsergym"
RUN_NAME = "openenv-browsergym"
PROJECT = "openenv-browsergym"
SPACE_ID = "OpenEnv-BrowserGym"
USE_TRACKIO = False
# Environment params for rollout builder
DOCKER_IMAGE = "browsergym-env:latest"
MINIWOB_URL = "http://172.17.0.1:8888/miniwob"
TASK_LIST = ["click-test", "click-button"]

# PEFT / LoRA config
PEFT_RANK = int(cfg.get("peft", {}).get("lora_rank", 32))
PEFT_ALPHA = float(cfg.get("peft", {}).get("lora_alpha", 32))
PEFT_TARGET = cfg.get("peft", {}).get("target_modules", "all-linear")

# Fireworks hotload config
FW_ACCOUNT_ID = "pyroworks"
FW_DEPLOYMENT_ID = "h6mxm330"
FW_MODEL_ID_PREFIX = "openenv-browsergym"
FW_HOTLOAD_EVERY_N = 1
FW_PROBE_AFTER = True
FW_BASE_MODEL_RESOURCE = "accounts/fireworks/models/qwen3-8b"


# ---------------------------------------------------------------------------
# User Callables
# ---------------------------------------------------------------------------

ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)


def _build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def _as_scalar(x: Any) -> Any:
    try:
        return x.item()
    except Exception:
        return x


def _extract_goal_url_title(observation: Any) -> Tuple[str, str, str]:
    """
    Returns (goal, url, title).
    Goal is taken from observation.goal, or metadata.browsergym_obs.goal,
    or goal_object text, or last user chat message.
    """
    goal = getattr(observation, "goal", "") or ""
    url = getattr(observation, "url", "") or ""
    title = ""

    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}

    if not goal:
        goal = obs_dict.get("goal") or ""
    if not goal:
        goal_object = obs_dict.get("goal_object")
        # goal_object looks like tuple of dicts with {'type':'text','text':'...'}
        if isinstance(goal_object, (list, tuple)) and goal_object:
            for item in goal_object:
                if isinstance(item, dict) and item.get("type") == "text":
                    goal = str(item.get("text", "")).strip()
                    if goal:
                        break
    if not goal:
        # Fallback: last user message in chat_messages
        chat = obs_dict.get("chat_messages")
        if isinstance(chat, (list, tuple)) and chat:
            for msg in reversed(chat):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    goal = str(msg.get("message", "")).strip()
                    if goal:
                        break

    if not url:
        url = obs_dict.get("url") or ""

    titles = obs_dict.get("open_pages_titles") or ()
    active_idx = _as_scalar(obs_dict.get("active_page_index"))
    try:
        active_idx = int(active_idx)
    except Exception:
        active_idx = 0
    if isinstance(titles, (list, tuple)) and 0 <= active_idx < len(titles):
        title = titles[active_idx] or ""

    return goal, url, title


def _get_bg_obs(observation: Any) -> dict:
    metadata = getattr(observation, "metadata", {}) or {}
    return metadata.get("browsergym_obs", {}) or {}


def _elapsed_time_str(obs_dict: dict) -> str:
    et = obs_dict.get("elapsed_time")
    try:
        et = et.item() if hasattr(et, "item") else float(et)
        return f"{et:.2f}s"
    except Exception:
        return "-"


def _extract_clickable_elements(observation) -> List[str]:
    """
    Collect BrowserGym element IDs that can be clicked from observation.metadata.
    Returns formatted lines for inclusion in the prompt.
    """
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = obs_dict.get("focused_element_bid")

    # Build a map from browsergym_id (BID) -> (role, name)
    bid_to_desc = {}
    try:
        nodes = axtree_object.get("nodes") or []
        for node in nodes:
            bid = node.get("browsergym_id")
            if bid is None:
                continue
            role = ""
            name = ""
            rf = node.get("role") or {}
            if isinstance(rf, dict):
                role = str(rf.get("value", "")).strip()
            nf = node.get("name") or {}
            if isinstance(nf, dict):
                name = str(nf.get("value", "")).strip()
            bid_to_desc[str(bid)] = (role, name)
    except Exception:
        pass

    lines: List[str] = []
    # Keep a stable ordering for readability
    for bid in sorted(extra_props.keys(), key=lambda x: str(x)):
        props = extra_props[bid] or {}
        if not props.get("clickable"):
            continue
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(v) for v in bbox) if bbox else "?"
        role, name = bid_to_desc.get(str(bid), ("", ""))
        focus_tag = " [FOCUSED]" if (str(bid) == str(focused_bid)) else ""
        rn = (role or "-")
        if name:
            rn = f"{rn} | {name}"
        vis = props.get("visibility")
        vis_str = f"{vis:.2f}" if isinstance(vis, (int, float)) else str(vis) if vis is not None else "?"
        lines.append(f"- BID {bid}{focus_tag}: {rn} | bbox({bbox_str}) | visibility={vis_str}")
    return lines


def _rank_clickables_by_goal(observation: Any, goal: str, top_n: int = 8) -> Tuple[List[str], Optional[str]]:
    """
    Heuristically rank clickable BIDs by goal match.
    Returns (ranked_lines, recommended_bid).
    """
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    goal_lc = (goal or "").lower().strip()

    # Build details similar to _extract_clickable_elements, but keep structured tuples
    extra_props = obs_dict.get("extra_element_properties", {}) or {}
    axtree_object = obs_dict.get("axtree_object") or {}
    focused_bid = str(obs_dict.get("focused_element_bid") or "")
    bid_to_desc = {}
    try:
        nodes = axtree_object.get("nodes") or []
        for node in nodes:
            bid = node.get("browsergym_id")
            if bid is None:
                continue
            role = ""
            name = ""
            rf = node.get("role") or {}
            if isinstance(rf, dict):
                role = str(rf.get("value", "")).strip()
            nf = node.get("name") or {}
            if isinstance(nf, dict):
                name = str(nf.get("value", "")).strip()
            bid_to_desc[str(bid)] = (role, name)
    except Exception:
        pass

    scored: List[Tuple[float, str, str, str, str]] = []
    for bid_key in sorted(extra_props.keys(), key=lambda x: str(x)):
        props = extra_props[bid_key] or {}
        if not props.get("clickable"):
            continue
        role, name = bid_to_desc.get(str(bid_key), ("", ""))
        name_lc = (name or "").lower()
        # Simple score: substring match + role=button bonus + focused bonus + visibility
        score = 0.0
        if goal_lc and name_lc and (goal_lc in name_lc or name_lc in goal_lc):
            score += 2.0
        if (role or "").lower() == "button":
            score += 1.0
        if str(bid_key) == focused_bid:
            score += 0.5
        vis = props.get("visibility")
        try:
            vis_f = float(vis)
            score += max(0.0, min(1.0, vis_f))
        except Exception:
            pass
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(v) for v in bbox) if bbox else "?"
        rn = (role or "-")
        if name:
            rn = f"{rn} | {name}"
        scored.append((score, str(bid_key), rn, bbox_str, f"{vis:.2f}" if isinstance(vis, (int, float)) else str(vis) if vis is not None else "?"))

    scored.sort(key=lambda t: t[0], reverse=True)
    lines: List[str] = []
    recommended = scored[0][1] if scored else None
    for idx, (score, bid, rn, bbox_str, vis_str) in enumerate(scored[:top_n], start=1):
        lines.append(f"{idx}. BID {bid}: score={score:.2f} | {rn} | bbox({bbox_str}) | visibility={vis_str}")
    return lines, recommended


def _extract_screenshot_image(observation) -> Image.Image | None:
    """
    Convert observation.screenshot (HWC uint8) into PIL Image if present.
    """
    screenshot = getattr(observation, "screenshot", None)
    if screenshot is None:
        return None
    try:
        arr = np.array(screenshot, dtype=np.uint8)
        return Image.fromarray(arr)
    except Exception:
        return None


def prompt_builder(observation: Any, step: int, history: List[str]) -> Any:
    """
    Build the user message content for the LLM.
    Return a string (text-only). We intentionally skip multimodal here.
    """
    goal, url, title = _extract_goal_url_title(observation)
    url = url or "(unknown)"
    error_note = "Yes" if getattr(observation, "last_action_error", False) else "No"

    # Clickable BIDs
    clickables = _extract_clickable_elements(observation)
    clickable_block = "\n".join(clickables) if clickables else "(none detected)"
    ranked_clickables, recommended_bid = _rank_clickables_by_goal(observation, goal, top_n=10)
    ranked_block = "\n".join(ranked_clickables) if ranked_clickables else "(none)"

    # Build textual prompt
    text = getattr(observation, "text", "") or ""
    text = text[:MAX_TOKENS * 20]  # constrain to a reasonable size
    # MiniWoB extras: AX tree and pruned HTML (when available)
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    axtree_text = (
        getattr(observation, "axtree_txt", None)
        or getattr(observation, "ax_tree_txt", None)
        or obs_dict.get("axtree_txt")
        or obs_dict.get("ax_tree_txt")
        or ""
    )
    pruned_html = (
        getattr(observation, "pruned_html", None)
        or obs_dict.get("pruned_html")
        or ""
    )
    axtree_text = str(axtree_text)[:MAX_TOKENS * 30]
    pruned_html = str(pruned_html)[:MAX_TOKENS * 30]
    # Focus info
    focused_bid = obs_dict.get("focused_element_bid") or ""
    elapsed_str = _elapsed_time_str(obs_dict)
    last_action = obs_dict.get("last_action") or ""

    user_prompt = (
        f"Step: {step}\n"
        f"Goal: {goal}\n"
        f"Current URL: {url}\n"
        f"Title: {title}\n"
        f"Elapsed: {elapsed_str}\n"
        f"Previous steps:\n{_build_history_lines(history)}\n"
        f"Last action: {last_action}\n"
        f"Last action error: {error_note}\n"
        f"Focused BID: {focused_bid}\n\n"
        f"Clickable elements (BID: role | name | bbox | visibility):\n{clickable_block}\n\n"
        f"Ranked clickable candidates (best first):\n{ranked_block}\n"
        f"Recommended BID: {recommended_bid or '(none)'}\n\n"
        "Instructions:\n"
        "- Choose the most relevant clickable BID to achieve the goal.\n"
        "- Prefer role=button or elements whose name matches the goal.\n"
        "- Reply with a single action, e.g., click('13') or noop().\n\n"
        f"Page excerpt:\n{text}\n\n"
        f"AXTree excerpt:\n{axtree_text}\n\n"
        f"Pruned HTML excerpt:\n{pruned_html}\n\n"
        "Reply with exactly one BrowserGym action string."
    ).strip()

    return user_prompt


def action_parser(response_text: str) -> BrowserGymAction:
    """
    Parse a BrowserGymAction from the LLM response text.
    """
    if not response_text:
        return BrowserGymAction(action_str="noop()")

    # Prefer first line that matches the action pattern
    for raw in response_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = ACTION_PATTERN.search(line)
        if m:
            return BrowserGymAction(action_str=re.sub(r"\s+", " ", m.group(0)))

    # Fallback: search whole response
    m = ACTION_PATTERN.search(response_text)
    if m:
        return BrowserGymAction(action_str=parsed)

    return BrowserGymAction(action_str="noop()")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Processor/tokenizer for TRL
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    max_prompt_len = 512

    # Minimal dataset: prompt is typically constant; env observations drive the loop
    dataset = Dataset.from_dict(
        {"prompt": ["You are a web navigation agent."] * DATASET_SIZE}
    )

    # Build rollout_func using eval_protocol integration
    rollout_func = create_openenv_rollout_func(
        env_factory=None,  # build default factory from params below
        prompt_builder=prompt_builder,
        action_parser=action_parser,
        model=INFER_MODEL_ID,
        max_steps=MAX_STEPS,
        completion_params={
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            **({"top_p": TOP_P} if TOP_P is not None else {}),
            "system_prompt": (
                "You control a web browser.\n"
                "Commands:\n"
                "- noop()\n"
                "- click('<BID>')\n"
                "- type('selector', 'text to enter')\n"
                "- fill('selector', 'text to enter')\n"
                "- send_keys('Enter')\n"
                "- scroll('down')\n"
                "Reply with exactly one BrowserGym action string."
            ),
        },
        concurrency=PER_DEVICE_BATCH_SIZE,
        tasks=TASK_LIST,
        miniwob_url=MINIWOB_URL,
        docker_image=DOCKER_IMAGE,
    )

    # TRL configuration
    # Enable TensorBoard by default; optionally also Trackio if enabled
    _report_to = ["tensorboard"]
    if USE_TRACKIO:
        _report_to.append("trackio")
    _report_kwargs = {
        "report_to": _report_to,
        "trackio_space_id": SPACE_ID if USE_TRACKIO else None,
        "logging_dir": str(Path(OUTPUT_DIR) / "tensorboard"),
    }
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        logging_steps=1,
        run_name=RUN_NAME,
        project=PROJECT,
        max_completion_length=MAX_TOKENS,
        max_prompt_length=max_prompt_len,
        learning_rate=LEARNING_RATE,
        reward_weights=[1.0],
        **_report_kwargs,
    )
    # Define a simple reward (sum of per-step rewards)
    def reward_sum(completions: List[str], **kwargs) -> List[float]:
        rewards = kwargs.get("step_rewards", [])
        return [float(np.sum(r)) if r else 0.0 for r in rewards]

    # LoRA PEFT configuration for local training
    peft_config = LoraConfig(
        r=PEFT_RANK,
        lora_alpha=PEFT_ALPHA,
        target_modules=None if PEFT_TARGET != "all-linear" else None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Create a generic OpenEnv-enabled trainer class from any TRL trainer (default: GRPOTrainer)
    OpenEnvTrainer = create_openenv_trainer_class(_BaseTRLTrainer)
    # Instantiate trainer (Fireworks hot-load wired internally via kwargs)
    trainer = OpenEnvTrainer(
        model=MODEL_ID,
        processing_class=processor,
        reward_funcs=[reward_sum],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        openenv_rollout_func=rollout_func,
        fireworks_account_id=FW_ACCOUNT_ID or None,
        fireworks_deployment_id=FW_DEPLOYMENT_ID or None,
        fireworks_base_model_resource=(FW_BASE_MODEL_RESOURCE or MODEL_ID),
        fireworks_model_id_prefix=FW_MODEL_ID_PREFIX,
        fireworks_hotload_every_n_steps=FW_HOTLOAD_EVERY_N,
        fireworks_probe_after_hotload=FW_PROBE_AFTER,
    )

    trainer.train()


if __name__ == "__main__":
    main()


