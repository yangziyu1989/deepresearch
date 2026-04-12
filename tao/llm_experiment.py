"""Minimal runnable LLM fine-tuning utilities for Tao experiment tasks."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from tao.experiment_tasks import resolve_dataset_info, resolve_model_id

try:
    import torch.nn as nn
except Exception:  # pragma: no cover - used only when experiment deps are absent locally
    class _FallbackModule:
        pass

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()


def _require_training_libs() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Experiment dependencies are missing. Install with: pip install -e '.[experiment]'"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _target_modules() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def _format_example(example: dict, tokenizer: Any) -> str:
    if example.get("text"):
        return str(example["text"])

    if example.get("messages") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    if example.get("conversations") and getattr(tokenizer, "chat_template", None):
        messages = []
        for turn in example["conversations"]:
            role = turn.get("from") or turn.get("role") or "user"
            content = turn.get("value") or turn.get("content") or ""
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            messages.append({"role": role, "content": content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    instruction = example.get("instruction") or example.get("prompt") or ""
    extra_input = example.get("input") or ""
    output = example.get("output") or example.get("response") or example.get("completion") or ""
    if extra_input:
        instruction = f"{instruction}\n\nInput:\n{extra_input}".strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}".strip()


def _tokenize_dataset(raw_dataset: Any, tokenizer: Any, max_length: int) -> Any:
    def _tokenize(batch: dict) -> dict:
        texts = [_format_example({k: batch[k][i] for k in batch}, tokenizer) for i in range(len(next(iter(batch.values()))))]
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = [
            [token if mask else -100 for token, mask in zip(ids, attn)]
            for ids, attn in zip(tokens["input_ids"], tokens["attention_mask"])
        ]
        return tokens

    keep_cols = list(raw_dataset.column_names)
    return raw_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=keep_cols,
        desc="Tokenizing SFT dataset",
    )


class _CausalLMCollator:
    def __call__(self, features: list[dict]) -> dict:
        libs = _require_training_libs()
        torch = libs["torch"]
        batch = {}
        for key in features[0]:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch


def _load_dataset_splits(task: dict, tokenizer: Any) -> tuple[Any, Any]:
    libs = _require_training_libs()
    load_dataset = libs["load_dataset"]

    dataset_info = resolve_dataset_info(task["dataset"])
    dataset_id = task.get("dataset_id") or dataset_info["dataset_id"]
    split = task.get("dataset_split") or dataset_info.get("split", "train")
    raw = load_dataset(dataset_id, split=split)
    raw = raw.shuffle(seed=42)

    train_examples = int(task.get("hyperparameters", {}).get("train_examples", len(raw)))
    eval_examples = int(task.get("hyperparameters", {}).get("eval_examples", max(100, min(500, len(raw) // 20))))

    selected = raw.select(range(min(len(raw), train_examples + eval_examples)))
    if len(selected) <= eval_examples:
        raise RuntimeError("Not enough examples to create train/eval split")

    train_dataset = selected.select(range(min(train_examples, len(selected) - eval_examples)))
    eval_dataset = selected.select(range(len(train_dataset), min(len(selected), len(train_dataset) + eval_examples)))

    max_length = int(task.get("hyperparameters", {}).get("sequence_length", 4096))
    return _tokenize_dataset(train_dataset, tokenizer, max_length), _tokenize_dataset(eval_dataset, tokenizer, max_length)


def _load_model_and_tokenizer(task: dict, routed: bool) -> tuple[Any, Any]:
    libs = _require_training_libs()
    torch = libs["torch"]
    AutoModelForCausalLM = libs["AutoModelForCausalLM"]
    AutoTokenizer = libs["AutoTokenizer"]
    BitsAndBytesConfig = libs["BitsAndBytesConfig"]
    prepare_model_for_kbit_training = libs["prepare_model_for_kbit_training"]
    LoraConfig = libs["LoraConfig"]
    get_peft_model = libs["get_peft_model"]

    model_id = resolve_model_id(task.get("model", ""))
    quantization = task.get("hyperparameters", {}).get("quantization", "")
    use_4bit = quantization == "nf4"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype if not use_4bit else None,
        device_map="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    method = str(task.get("code_template", "")).lower()
    use_dora = "dora" in method
    lora_rank = int(task.get("hyperparameters", {}).get("lora_rank", 64))
    lora_alpha = int(task.get("hyperparameters", {}).get("lora_alpha", 128))
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_target_modules(),
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora_cfg)

    if routed:
        apply_mlp_token_routing(
            model,
            route_fraction=float(task.get("hyperparameters", {}).get("route_fraction", 0.25)),
            route_start_layer=int(task.get("hyperparameters", {}).get("route_start_layer", -1)),
        )

    return model, tokenizer


class RoutedMLPDecoderLayer(nn.Module):
    """Wrap a decoder layer so only a routed token subset goes through the heavy MLP."""

    def __init__(self, base_layer: Any, route_fraction: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.route_fraction = max(0.0, min(route_fraction, 1.0))
        self.last_route_fraction = 1.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_layer, name)

    def _route_mask(self, hidden_states: Any) -> Any:
        torch = _require_training_libs()["torch"]
        batch, seq_len, _ = hidden_states.shape
        if self.route_fraction >= 1.0 or seq_len <= 1:
            mask = torch.ones((batch, seq_len), dtype=torch.bool, device=hidden_states.device)
            self.last_route_fraction = 1.0
            return mask

        scores = hidden_states.norm(dim=-1)
        k = max(1, int(seq_len * self.route_fraction))
        topk = scores.topk(k=k, dim=1).indices
        mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=hidden_states.device)
        mask.scatter_(1, topk, True)
        self.last_route_fraction = float(mask.float().mean().item())
        return mask

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        residual = hidden_states
        normed = self.base_layer.input_layernorm(hidden_states)
        attn_outputs = self.base_layer.self_attn(normed, *args, **kwargs)
        if isinstance(attn_outputs, tuple):
            attn_hidden = attn_outputs[0]
            rest = attn_outputs[1:]
        else:
            attn_hidden = attn_outputs
            rest = ()

        hidden_states = residual + attn_hidden
        residual = hidden_states
        normed = self.base_layer.post_attention_layernorm(hidden_states)
        route_mask = self._route_mask(normed)

        torch = _require_training_libs()["torch"]
        mlp_out = torch.zeros_like(normed)
        selected = normed[route_mask]
        if selected.numel() > 0:
            mlp_out[route_mask] = self.base_layer.mlp(selected)

        hidden_states = residual + mlp_out
        return (hidden_states,) + rest


def apply_mlp_token_routing(model: Any, route_fraction: float, route_start_layer: int = -1) -> None:
    """Patch later decoder layers with routed-MLP wrappers."""
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        layers = model.base_model.model.layers
    if layers is None:
        raise RuntimeError("Could not find decoder layers for routed PEFT patching")

    start_idx = route_start_layer if route_start_layer >= 0 else max(0, len(layers) // 2)
    for idx in range(start_idx, len(layers)):
        layers[idx] = RoutedMLPDecoderLayer(layers[idx], route_fraction)


def collect_route_stats(model: Any) -> dict[str, float]:
    """Collect average route fractions from routed decoder layers."""
    stats: list[float] = []
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        layers = model.base_model.model.layers
    if layers is not None:
        for layer in layers:
            if hasattr(layer, "last_route_fraction"):
                stats.append(float(layer.last_route_fraction))
    if not stats:
        return {}
    return {
        "avg_route_fraction": sum(stats) / len(stats),
        "min_route_fraction": min(stats),
        "max_route_fraction": max(stats),
    }


def run_training_task(task: dict, workspace_root: str | Path, routed: bool = False) -> dict:
    """Run one dense or routed PEFT fine-tuning task."""
    libs = _require_training_libs()
    torch = libs["torch"]
    Trainer = libs["Trainer"]
    TrainingArguments = libs["TrainingArguments"]

    workspace_root = Path(workspace_root)
    task_id = task["id"]
    results_dir = workspace_root / "exp" / "results" / task_id
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    pid_file = workspace_root / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    model, tokenizer = _load_model_and_tokenizer(task, routed=routed)
    train_dataset, eval_dataset = _load_dataset_splits(task, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(results_dir / "hf_output"),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(task["hyperparameters"].get("batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(task["hyperparameters"].get("grad_accum_steps", 1)),
        learning_rate=float(task["hyperparameters"].get("learning_rate", 2e-4)),
        num_train_epochs=float(task["hyperparameters"].get("epochs", 1)),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=int((task.get("seeds") or [42])[0]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_CausalLMCollator(),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed = time.time() - start
    total_tokens = len(train_dataset) * int(task["hyperparameters"].get("sequence_length", 4096))
    route_stats = collect_route_stats(model)
    results = {
        "task_id": task_id,
        "status": "success",
        "elapsed_sec": elapsed,
        "model_id": resolve_model_id(task.get("model", "")),
        "dataset": resolve_dataset_info(task["dataset"])["dataset_id"],
        "routed": routed,
        "metrics": {
            "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
            "perplexity": float(torch.exp(torch.tensor(eval_metrics["eval_loss"])).item()) if eval_metrics.get("eval_loss") is not None else None,
            "tokens_per_second": float(total_tokens / max(elapsed, 1e-6)),
            "peak_vram_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0,
            **route_stats,
        },
        "hyperparameters": task.get("hyperparameters", {}),
    }

    (results_dir / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (workspace_root / f"{task_id}_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (workspace_root / f"{task_id}_DONE").write_text(json.dumps(results, indent=2), encoding="utf-8")
    progress = {
        "task_id": task_id,
        "status": "done",
        "elapsed_sec": elapsed,
        "tokens_per_second": results["metrics"]["tokens_per_second"],
        **route_stats,
    }
    (workspace_root / f"{task_id}_PROGRESS.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    if pid_file.exists():
        pid_file.unlink()
    return results
