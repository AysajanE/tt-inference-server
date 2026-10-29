# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3.1-Flash reuses the MiniMax-M3 eval tasks in its plan of record,
plus DeepSWE, with the model card's published scores and #5470's GPU references."""

from dataclasses import replace

from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
)
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    DEEPSWE_TEST_WORKER_ENV,
)

M3 = "MiniMaxAI/MiniMax-M3"
M31_FLASH = "MiniMaxAI/MiniMax-M3.1-Flash"
CARD = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash"
ISSUE = "https://github.com/tenstorrent/tt-inference-server/issues/5470"
M3_TASKS = {
    "r1_gpqa_diamond",
    "longbench2_generate",
    "terminal_bench_2_1",
    "terminal_bench_4",
    "automation_bench",
}
PUBLISHED = {"automation_bench": 62.8, "terminal_bench_4": 38.1, "deepswe": 73.7}
GPU_REFERENCE = {
    "r1_gpqa_diamond": 87.88,
    "longbench2_generate": 63.49,
    "automation_bench": 61.33,
}


def _tasks(repo):
    return {t.task_name: t for t in ALL_EVAL_CONFIGS[repo].tasks}


def test_m31_flash_task_set():
    assert _tasks(M31_FLASH).keys() == M3_TASKS | {"deepswe"}


def test_m31_flash_m3_tasks_match_m3_apart_from_targets_effort_and_concurrency():
    m3 = _tasks(M3)
    for name, task in _tasks(M31_FLASH).items():
        if name not in M3_TASKS:
            continue
        expected = replace(m3[name], score=task.score, request_body=task.request_body)
        if task.agentic_eval_config is not None:
            cfg = task.agentic_eval_config
            fields = {
                "n_concurrent_trials": cfg.n_concurrent_trials,
                "agent_kwargs": cfg.agent_kwargs,
            }
            for field in ("override_cpus", "override_memory_mb"):
                if hasattr(cfg, field):
                    fields[field] = getattr(cfg, field)
            expected = replace(
                expected,
                agentic_eval_config=replace(expected.agentic_eval_config, **fields),
            )
        assert expected == task, name


def test_m31_flash_asks_for_max_effort_on_every_task():
    tasks = _tasks(M31_FLASH)
    for name, task in tasks.items():
        if name == "automation_bench":
            effort = task.agentic_eval_config.agent_kwargs["reasoning_effort"]
        else:
            effort = task.request_body["reasoning_effort"]
        assert effort == "max", name
    m3 = _tasks(M3)
    for name in M3_TASKS:
        assert m3[name].request_body == {
            k: v for k, v in tasks[name].request_body.items() if k != "reasoning_effort"
        }, name
        if m3[name].agentic_eval_config is not None:
            assert (
                "reasoning_effort" not in m3[name].agentic_eval_config.agent_kwargs
            ), name


def test_m31_flash_agentic_concurrency_fits_kix1():
    # Sized to the KIX1 shield-evals pool (3 x 64 CPUs): trials that don't fit fail
    # Harbor's pod-ready timeout. M3 keeps its IAD-era values.
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    expected = {
        "terminal_bench_2_1": (62, 8),
        "terminal_bench_4": (40, 8),
        "automation_bench": (62, 62),
    }
    for name, (flash_n, m3_n) in expected.items():
        assert flash[name].agentic_eval_config.n_concurrent_trials == flash_n, name
        assert m3[name].agentic_eval_config.n_concurrent_trials == m3_n, name


def test_m31_flash_terminal_bench_2_1_uses_task_declared_resources():
    # M3's 8 CPU / 32 GiB override fits 7 trials per KIX1 node; the tasks
    # themselves mostly declare 1 CPU / 2 GiB.
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    cfg = flash["terminal_bench_2_1"].agentic_eval_config
    assert cfg.override_cpus is None and cfg.override_memory_mb is None
    assert m3["terminal_bench_2_1"].agentic_eval_config.override_cpus == 8


def test_m31_flash_terminal_bench_runs_terminus_2():
    for name in ("terminal_bench_2_1", "terminal_bench_4"):
        assert _tasks(M31_FLASH)[name].agentic_eval_config.agent == "terminus-2"


def test_m31_flash_deepswe_reaches_the_server_and_caps_test_workers():
    cfg = _tasks(M31_FLASH)["deepswe"].agentic_eval_config
    assert cfg.dataset == "datacurve/deep-swe"
    assert cfg.allow_agent_endpoint
    assert cfg.agent_kwargs["version"] == "2.4.5"
    assert cfg.n_concurrent_trials == 62
    assert cfg.environment_env == DEEPSWE_TEST_WORKER_ENV
    assert cfg.verifier_env == DEEPSWE_TEST_WORKER_ENV


def test_m31_flash_targets():
    for name, task in _tasks(M31_FLASH).items():
        score = task.score
        assert score.published_score == PUBLISHED.get(name), name
        assert score.published_score_ref == (CARD if name in PUBLISHED else None)
        assert score.gpu_reference_score == GPU_REFERENCE.get(name), name
        if name in GPU_REFERENCE:
            assert score.gpu_reference_score_ref.startswith(ISSUE), name
        else:
            assert score.gpu_reference_score_ref is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS
