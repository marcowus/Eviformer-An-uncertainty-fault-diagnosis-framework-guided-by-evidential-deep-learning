"""Command line interface for triggering the secondary LLM review pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..feature_engineering.builder import ModelOutputs, build_feature_packet
from ..helpers import get_device
from ..llm.client import EchoLLMClient, LLMClient, LLMClientError
from ..llm.reviewer import ReviewVerdict, run_llm_review
from ..losses import relu_evidence
from ..uncertainty.triage import TriageThresholds, should_escalate

try:
    from ..data import data_transforms, load_cwru_dataset
except ImportError:  # pragma: no cover - optional dependency for synthetic runs
    data_transforms = None
    load_cwru_dataset = None

try:
    from ..MCSwinT import mcswint
except ImportError:  # pragma: no cover - optional
    mcswint = None

try:
    from ..VIT import vit_middle_patch16
except ImportError:  # pragma: no cover - optional
    vit_middle_patch16 = None

try:
    from ..lenet import LeNet
except ImportError:  # pragma: no cover - optional
    LeNet = None

class SyntheticSignalDataset(Dataset):
    def __init__(self, num_samples: int, signal_length: int, num_classes: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.signals = rng.normal(size=(num_samples, signal_length)).astype(np.float32)
        self.labels = rng.integers(0, num_classes, size=num_samples, dtype=np.int64)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int):
        return self.signals[index], int(self.labels[index])


ARCH_REGISTRY = {
    "mcswint": lambda num_classes: mcswint(in_channel=1, out_channel=num_classes) if mcswint else None,
    "vit": lambda num_classes: vit_middle_patch16(data_size=1024, in_c=1, num_cls=num_classes, h_args=[256, 128, 64, 32])
    if vit_middle_patch16
    else None,
    "lenet": lambda num_classes: LeNet(dropout=False) if LeNet else None,
}


def _build_model(arch: str, num_classes: int) -> Optional[torch.nn.Module]:
    factory = ARCH_REGISTRY.get(arch.lower())
    if factory is None:
        raise ValueError(f"Unsupported architecture '{arch}'. Options: {list(ARCH_REGISTRY)}")
    model = factory(num_classes)
    if model is None:
        raise RuntimeError(
            f"Architecture '{arch}' is not available in the current environment."
        )
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def _prepare_dataloader(args: argparse.Namespace) -> DataLoader:
    if args.synthetic:
        dataset = SyntheticSignalDataset(args.synthetic_samples, args.signal_length, args.num_classes)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if load_cwru_dataset is None:
        raise RuntimeError("CWRU dataset utilities are unavailable; enable --synthetic to run without data.")

    train_ds, val_ds = load_cwru_dataset(args.data_dir, args.normalize, test=False)
    dataset = val_ds if not args.use_train_split else train_ds
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def _prepare_client(args: argparse.Namespace) -> LLMClient:
    if args.llm_backend == "echo" or args.llm_backend is None:
        return EchoLLMClient()
    return LLMClient(base_url=args.llm_base_url, api_key=args.llm_api_key, model=args.llm_model)


def _tensor_from_seq(seq: Any) -> torch.Tensor:
    if isinstance(seq, torch.Tensor):
        return seq.float()
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return torch.from_numpy(arr).float()


def _raw_signal_from_seq(seq: Any) -> np.ndarray:
    arr = np.asarray(seq)
    return np.squeeze(arr).astype(np.float64)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_primary_metrics(
    alpha: torch.Tensor,
    probabilities: torch.Tensor,
    uncertainties: torch.Tensor,
    labels: torch.Tensor,
    *,
    idx: int,
) -> ModelOutputs:
    predicted_label = int(torch.argmax(probabilities[idx]).item())
    return ModelOutputs(
        predicted_label=predicted_label,
        probabilities=probabilities[idx].detach().cpu().numpy().tolist(),
        uncertainty=float(uncertainties[idx].detach().cpu().item()),
        alpha=alpha[idx].detach().cpu().numpy().tolist(),
        true_label=int(labels[idx].item()) if labels is not None else None,
        evidence=float((alpha[idx].detach().cpu().numpy() - 1.0).sum()),
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device() if not args.cpu else torch.device("cpu")
    dataloader = _prepare_dataloader(args)

    if not args.skip_model:
        model = _build_model(args.arch, args.num_classes)
        model = model.to(device)
        if args.checkpoint:
            _load_checkpoint(model, Path(args.checkpoint), device)
        model.eval()
    else:
        model = None

    llm_client = _prepare_client(args)
    thresholds = TriageThresholds(
        max_probability=args.max_probability,
        min_uncertainty=args.min_uncertainty,
        min_evidence=args.min_evidence,
        min_alpha_sum=args.min_alpha_sum,
    )

    records: List[Dict[str, Any]] = []

    rng = np.random.default_rng(args.synthetic_seed)

    for batch in dataloader:
        inputs, labels = batch
        batch_size = len(labels)
        tensor_inputs = _tensor_from_seq(inputs)
        if tensor_inputs.ndim == 2:
            tensor_inputs = tensor_inputs.unsqueeze(1)
        raw_signals = [
            _raw_signal_from_seq(inputs[i]) if isinstance(inputs, (list, tuple)) else _raw_signal_from_seq(tensor_inputs[i].cpu().numpy())
            for i in range(batch_size)
        ]

        if model is not None:
            tensor_inputs = tensor_inputs.to(device)
            with torch.no_grad():
                logits = model(tensor_inputs)
                evidence = relu_evidence(logits)
                alpha = evidence + 1
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
                uncertainties = args.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        else:
            probs_np = rng.uniform(size=(batch_size, args.num_classes))
            probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
            probs = torch.from_numpy(probs_np).float()
            alpha = probs * args.synthetic_alpha_scale + 1.0
            uncertainties = torch.full((batch_size, 1), args.synthetic_uncertainty, dtype=torch.float32)

        if isinstance(labels, torch.Tensor):
            labels_tensor = labels
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        for idx in range(batch_size):
            prob_vec = probs[idx].detach().cpu().numpy()
            unc_val = float(uncertainties[idx].detach().cpu().item())
            alpha_vec = alpha[idx].detach().cpu().numpy()
            escalate = should_escalate(
                probabilities=prob_vec,
                uncertainty=unc_val,
                alpha=alpha_vec,
                thresholds=thresholds,
            )

            model_outputs = _collect_primary_metrics(
                alpha=alpha,
                probabilities=probs,
                uncertainties=uncertainties,
                labels=labels_tensor,
                idx=idx,
            )

            record: Dict[str, Any] = {
                "index": len(records),
                "label": int(labels_tensor[idx].item()),
                "primary": model_outputs.to_serializable(),
                "escalated": escalate,
            }

            if escalate:
                feature_packet = build_feature_packet(
                    raw_signals[idx],
                    sampling_rate=args.sampling_rate,
                    model_outputs=model_outputs,
                )
                try:
                    verdict: ReviewVerdict = run_llm_review(
                        feature_packet,
                        model_outputs.to_serializable(),
                        client=llm_client,
                        language=args.language,
                        temperature=args.temperature,
                    )
                    record["secondary"] = {
                        "final_diagnosis": verdict.final_diagnosis,
                        "confidence": verdict.confidence,
                        "conflict": verdict.conflict_with_primary,
                        "rationale": verdict.rationale,
                        "checks": verdict.checks,
                        "maintenance": verdict.maintenance,
                    }
                    record["feature_packet"] = feature_packet
                except LLMClientError as exc:
                    record["secondary_error"] = str(exc)
            records.append(record)

    if args.output:
        _ensure_dir(Path(args.output).parent)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(records, fp, ensure_ascii=False, indent=2)

    return {"num_records": len(records), "escalated": sum(1 for r in records if r.get("escalated"))}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secondary LLM review pipeline")
    parser.add_argument("--data-dir", type=str, default="D:/CWRU", help="Path to raw CWRU dataset root")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint file for the primary model")
    parser.add_argument("--arch", type=str, default="mcswint", help="Primary model architecture")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--normalize", type=str, default="0-1")
    parser.add_argument("--sampling-rate", type=float, default=12000.0)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--use-train-split", action="store_true", help="Use the training split instead of validation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for inference")
    parser.add_argument("--skip-model", action="store_true", help="Skip model inference and use synthetic predictions")

    # Synthetic data options
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of loading the dataset")
    parser.add_argument("--synthetic-samples", type=int, default=4, help="Number of synthetic samples")
    parser.add_argument("--signal-length", type=int, default=1024)
    parser.add_argument("--synthetic-alpha-scale", type=float, default=50.0)
    parser.add_argument("--synthetic-uncertainty", type=float, default=0.35)
    parser.add_argument("--synthetic-seed", type=int, default=0)

    # Triage thresholds
    parser.add_argument("--max-probability", type=float, default=0.6)
    parser.add_argument("--min-uncertainty", type=float, default=0.25)
    parser.add_argument("--min-evidence", type=float, default=30.0)
    parser.add_argument("--min-alpha-sum", type=float, default=60.0)

    # LLM parameters
    parser.add_argument("--llm-backend", type=str, default="echo", help="Backend identifier (echo or openai-compatible)")
    parser.add_argument("--llm-base-url", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    stats = run_pipeline(args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
