from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from accelerate.utils import gather_object, broadcast_object_list

from eval_protocol.pytest.integrations.openenv_trl import create_openenv_rollout_func
try:
    from trl.examples.scripts.eval_protocol_gsm8k_grpo.hotload_callback import FireworksHotLoadCallback  # type: ignore
except Exception:
    from hotload_callback import FireworksHotLoadCallback  # type: ignore


class FireworksOpenEnvTrainer:
    """
    Thin wrapper that keeps TRL's GRPOConfig untouched and only configures:
    - inference: model_id and completion params
    - concurrency: per-rollout parallelism
    - environment: tasks/miniwob_url/docker_image via env_factory

    Usage:
        grpo_args = GRPOConfig(...)
        trainer = FireworksOpenEnvTrainer(
            args=grpo_args,
            model_id=\"fireworks_ai/accounts/...\",   # inference model string
            completion_params={\"temperature\": 0.2, \"max_tokens\": 128},
            concurrency=grpo_args.per_device_train_batch_size,
            tasks=[\"click-test\", \"click-button\"],
            miniwob_url=\"http://172.17.0.1:8888/miniwob\",
            docker_image=\"browsergym-env:latest\",
            peft_lora_rank=32,
            peft_lora_alpha=32,
            fireworks_account_id=None,
            fireworks_deployment_id=None,
            fireworks_base_model_resource=None,
            fireworks_model_id_prefix=\"openenv-browsergym\",
            fireworks_hotload_every_n_steps=None,
            fireworks_probe_after_hotload=True,
        ).build(processor_model_id=\"Qwen/Qwen3-8B\")
        trainer.train()
    """

    def __init__(
        self,
        *,
        args: GRPOConfig,
        model_id: str,
        completion_params: Optional[Dict[str, Any]] = None,
        concurrency: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        miniwob_url: Optional[str] = None,
        docker_image: str = "browsergym-env:latest",
        # Optional PEFT
        peft_lora_rank: int = 32,
        peft_lora_alpha: float = 32.0,
        peft_target_modules: Optional[str] = "all-linear",
        # Optional Fireworks hot-load
        fireworks_account_id: Optional[str] = None,
        fireworks_deployment_id: Optional[str] = None,
        fireworks_base_model_resource: Optional[str] = None,
        fireworks_model_id_prefix: str = "openenv-browsergym",
        fireworks_hotload_every_n_steps: Optional[int] = None,
        fireworks_probe_after_hotload: bool = True,
    ):
        self.args = args
        self.infer_model_id = model_id
        self.completion_params = completion_params or {}
        self.concurrency = concurrency or args.per_device_train_batch_size
        self.tasks = tasks or []
        self.miniwob_url = miniwob_url
        self.docker_image = docker_image
        self.lora_rank = peft_lora_rank
        self.lora_alpha = peft_lora_alpha
        self.target_modules = peft_target_modules
        self.fw_acct = fireworks_account_id
        self.fw_deploy = fireworks_deployment_id
        self.fw_base_model = fireworks_base_model_resource
        self.fw_prefix = fireworks_model_id_prefix
        self.fw_every_n = fireworks_hotload_every_n_steps
        self.fw_probe = fireworks_probe_after_hotload

    def _env_factory(self):
        env_vars = {
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_HEADLESS": "true",
            "BROWSERGYM_VIEWPORT_WIDTH": "1280",
            "BROWSERGYM_VIEWPORT_HEIGHT": "720",
            "BROWSERGYM_TIMEOUT": "10000",
            "BROWSERGYM_OBS_AXTREE": "1",
            "BROWSERGYM_OBS_PRUNED_HTML": "1",
            "BROWSERGYM_RETURN_INFO": "1",
        }
        # Round-robin handled by the rollout processor; here we just pass defaults for container
        if self.tasks:
            # Let container pick task via env if provided by upstream
            env_vars["BROWSERGYM_TASK_NAME"] = str(self.tasks[0])
        if self.miniwob_url:
            env_vars["MINIWOB_URL"] = str(self.miniwob_url)
        return BrowserGymEnv.from_docker_image(self.docker_image, env_vars=env_vars)

    def build(self, *, processor_model_id: str, rollout_func=None) -> GRPOTrainer:
        processor = AutoProcessor.from_pretrained(processor_model_id)
        # Allow caller to supply their own rollout_func (preferred), or build one from integration params
        if rollout_func is None:
            rollout_func = create_openenv_rollout_func(
                env_factory=self._env_factory,
                prompt_builder=lambda obs, step, hist: str(getattr(obs, "text", "")) or "You are a web navigation agent.",
                action_parser=lambda txt: __import__("envs.browsergym_env", fromlist=["BrowserGymAction"]).BrowserGymAction(
                    action_str=(txt.strip().splitlines()[0] if txt else "noop()")
                ),
                model=self.infer_model_id,
                max_steps=getattr(self.args, "max_completion_length", 16),
                completion_params=self.completion_params,
                concurrency=self.concurrency,
            )

        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=None if self.target_modules == "all-linear" else None,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        callbacks = None
        if self.fw_acct and self.fw_deploy:
            callbacks = [
                FireworksHotLoadCallback(
                    account_id=self.fw_acct,
                    deployment_id=self.fw_deploy,
                    base_model=self.fw_base_model or processor_model_id,
                    model_id_prefix=self.fw_prefix,
                    hotload_every_n_steps=self.fw_every_n,
                    probe_after_hotload=self.fw_probe,
                )
            ]

        trainer = GRPOTrainer(
            model=processor_model_id,
            processing_class=processor,
            reward_funcs=[lambda completions, **kw: [float(sum(r or [])) for r in (kw.get("step_rewards") or [])]],
            train_dataset=None,  # caller should set, or replace trainer.train_dataset before train()
            args=self.args,
            peft_config=peft_config,
            callbacks=callbacks,
            rollout_func=rollout_func,
        )
        return trainer


class OpenEnvGRPOTrainer(GRPOTrainer):
    """
    Drop-in replacement mirroring TRL's GRPOTrainer signature, with optional Fireworks hot-load wiring.
    Usage:
        rollout_func = create_openenv_rollout_func(... all inference/env params ...)
        trainer = OpenEnvGRPOTrainer(
            model=...,
            processing_class=...,
            reward_funcs=[...],
            train_dataset=...,
            args=GRPOConfig(...),
            peft_config=...,
            rollout_func=rollout_func,               # pass directly like base GRPOTrainer
            # Optional Fireworks auto-hotload:
            fireworks_account_id="pyroworks",
            fireworks_deployment_id="h6mxm330",
            fireworks_base_model_resource="accounts/fireworks/models/qwen3-8b",
            fireworks_model_id_prefix="openenv-browsergym",
            fireworks_hotload_every_n_steps=1,
            fireworks_probe_after_hotload=True,
        )
    """

    def __init__(
        self,
        *args,
        openenv_rollout_func: Optional[Any] = None,
        fireworks_account_id: str | None = None,
        fireworks_deployment_id: str | None = None,
        fireworks_base_model_resource: str | None = None,
        fireworks_model_id_prefix: str = "openenv-browsergym",
        fireworks_hotload_every_n_steps: int | None = None,
        fireworks_probe_after_hotload: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.openenv_rollout_func = openenv_rollout_func

        # Wire Fireworks hot-load callback internally if configured
        if fireworks_account_id and fireworks_deployment_id:
            self.add_callback(
                FireworksHotLoadCallback(
                    account_id=fireworks_account_id,
                    deployment_id=fireworks_deployment_id,
                    base_model=fireworks_base_model_resource
                    or (getattr(self.processing_class, "name_or_path", None) or "unknown"),
                    model_id_prefix=fireworks_model_id_prefix,
                    hotload_every_n_steps=fireworks_hotload_every_n_steps,
                    probe_after_hotload=fireworks_probe_after_hotload,
                )
            )

    def _generate_single_turn(self, prompts: list):
        """
        Override generation to use the provided OpenEnv rollout function.
        Mirrors TRL's vLLM server path logic to dedupe prompts and broadcast outputs.
        """
        if self.openenv_rollout_func is None:
            # Fallback to base behavior if no custom rollout is provided
            return super()._generate_single_turn(prompts)

        all_prompts = gather_object(prompts)

        if self.accelerator.is_main_process:
            # Since 'prompts' contains 'num_generations' duplicates, first take unique prompts,
            # then generate num_generations outputs for each one.
            ordered_set_of_prompts = all_prompts[:: self.num_generations]

            output = self.openenv_rollout_func(
                ordered_set_of_prompts,
                self.args,
                self.processing_class,
            )
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            extra_fields = {k: v for k, v in output.items() if k not in required_keys}
            payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
        else:
            payload = None

        # Broadcast the completions from the main process to all processes, ensuring each process receives its slice.
        obj_list = [payload]
        broadcast_object_list(obj_list, from_process=0)
        all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

        # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
        all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        prompt_ids = all_prompt_ids[process_slice]
        completion_ids = all_completion_ids[process_slice]
        logprobs = all_logprobs[process_slice]

        # Slice extra fields dict-of-lists per process (extra fields are per-completion)
        extra_fields = {}
        for key, values in all_extra_fields.items():
            if isinstance(values, list):
                extra_fields[key] = values[process_slice]
            else:
                extra_fields[key] = values

        return prompt_ids, completion_ids, logprobs, extra_fields


def create_openenv_trainer_class(base_trainer_cls: Type[GRPOTrainer]) -> Type[GRPOTrainer]:
    """
    Create a subclass of any TRL online trainer that plugs in an OpenEnv rollout function.
    
    Usage:
        from trl import RLOOTrainer
        OpenEnvRLOOTrainer = create_openenv_trainer_class(RLOOTrainer)
        trainer = OpenEnvRLOOTrainer(
            openenv_rollout_func=my_rollout,
            model=..., processing_class=..., args=..., rollout_func=None,  # rollout_func not used by override
            ...
        )
    """

    class _OpenEnvAnyTrainer(base_trainer_cls):  # type: ignore[misc]
        def __init__(
            self,
            *args,
            openenv_rollout_func: Optional[Any] = None,
            fireworks_account_id: str | None = None,
            fireworks_deployment_id: str | None = None,
            fireworks_base_model_resource: str | None = None,
            fireworks_model_id_prefix: str = "openenv-browsergym",
            fireworks_hotload_every_n_steps: int | None = None,
            fireworks_probe_after_hotload: bool = True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.openenv_rollout_func = openenv_rollout_func

            # Optional Fireworks hot-load callback (works for any base trainer supporting callbacks)
            if fireworks_account_id and fireworks_deployment_id:
                self.add_callback(
                    FireworksHotLoadCallback(
                        account_id=fireworks_account_id,
                        deployment_id=fireworks_deployment_id,
                        base_model=fireworks_base_model_resource
                        or (getattr(self.processing_class, "name_or_path", None) or "unknown"),
                        model_id_prefix=fireworks_model_id_prefix,
                        hotload_every_n_steps=fireworks_hotload_every_n_steps,
                        probe_after_hotload=fireworks_probe_after_hotload,
                    )
                )

        def _generate_single_turn(self, prompts: list):
            # Only override if a rollout func is provided; otherwise defer to base implementation
            if self.openenv_rollout_func is None:
                return super()._generate_single_turn(prompts)  # type: ignore[attr-defined]

            all_prompts = gather_object(prompts)

            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts[:: self.num_generations]
                output = self.openenv_rollout_func(
                    ordered_set_of_prompts,
                    self.args,
                    self.processing_class,
                )
                required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
            else:
                payload = None

            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

            all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice]

            extra_fields = {}
            for key, values in all_extra_fields.items():
                if isinstance(values, list):
                    extra_fields[key] = values[process_slice]
                else:
                    extra_fields[key] = values

            return prompt_ids, completion_ids, logprobs, extra_fields

    _OpenEnvAnyTrainer.__name__ = f"OpenEnv{base_trainer_cls.__name__}"
    return _OpenEnvAnyTrainer

