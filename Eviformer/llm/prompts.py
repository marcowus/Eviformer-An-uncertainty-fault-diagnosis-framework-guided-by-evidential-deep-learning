"""Prompt templates for the secondary LLM reviewer."""
from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, Tuple


def _safe_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_feature_summary(packet: Dict[str, Any]) -> str:
    time_stats = packet.get("time", {})
    hist = packet.get("histogram", {})
    arma = packet.get("arma", {})

    top_freqs = hist.get("spectrum", {}).get("peak_frequencies", [])
    top_mags = hist.get("spectrum", {}).get("peak_magnitudes", [])

    lines = [
        "[时间域统计]",
        f"RMS: {_safe_float(time_stats.get('rms'))}",
        f"峰峰值: {_safe_float(time_stats.get('peak_to_peak'))}",
    ]
    lines.append(f"峭度: {_safe_float(time_stats.get('kurtosis'))}")
    lines.append(f"偏度: {_safe_float(time_stats.get('skewness'))}")

    if top_freqs:
        freq_lines = ", ".join(
            f"{freq:.1f}Hz({mag:.2f})" for freq, mag in zip(top_freqs, top_mags)
        )
    else:
        freq_lines = "无显著峰值"
    lines.append("[频域主峰] " + freq_lines)

    arma_order = arma.get("order", (0, 0))
    sigma2 = _safe_float(arma.get("sigma2"))
    lines.append(f"[ARMA 阶次] p={arma_order[0]}, q={arma_order[1]}, σ²={sigma2}")
    return "\n".join(lines)


def build_review_prompt(
    feature_packet: Dict[str, Any],
    primary_decision: Dict[str, Any],
    *,
    language: str = "zh",
) -> Tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the LLM call."""

    summary = _format_feature_summary(feature_packet)
    model_output = primary_decision
    feature_json = json.dumps(feature_packet, ensure_ascii=False, indent=2)

    if language.lower().startswith("zh"):
        system_prompt = textwrap.dedent(
            """
            你是一名旋转机械故障诊断专家，需要根据一次模型的输出和多种统计特征完成复核。
            请始终给出结构化的 JSON 结果，并在无法判断时明确说明原因。
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            ## 一次判别摘要
            模型预测类别: {model_output.get('predicted_label')}
            预测概率向量: {model_output.get('probabilities')}
            不确定度: {model_output.get('uncertainty')}
            Dirichlet 证据和 α: {model_output.get('alpha')}

            ## 关键特征概览
            {summary}

            ## 完整特征 JSON
            ```json
            {feature_json}
            ```

            ### 输出要求
            请以 JSON 形式回复，字段包括：
            - final_diagnosis: 字符串，判别的故障类型或"undetermined"
            - confidence: 0-100 的数值
            - rationale: 关键证据，列表或长文本均可
            - checks: 推荐的复核检测步骤数组
            - maintenance: 推荐的维修/保养措施数组
            如果需要驳回一次判别，请明确说明冲突原因。
            """
        ).strip()
    else:
        system_prompt = textwrap.dedent(
            """
            You are an expert vibration analyst tasked with validating fault diagnoses.
            Always respond with a JSON object and explain any ambiguity explicitly.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            ## Primary Model Summary
            Predicted label: {model_output.get('predicted_label')}
            Probability vector: {model_output.get('probabilities')}
            Uncertainty: {model_output.get('uncertainty')}
            Dirichlet evidence / alpha: {model_output.get('alpha')}

            ## Key Feature Highlights
            {summary}

            ## Full Feature JSON
            ```json
            {feature_json}
            ```

            ### Response schema
            Return a JSON payload with the following keys:
            - final_diagnosis (string)
            - confidence (0-100 number)
            - rationale (array of bullet explanations)
            - checks (array of recommended inspections)
            - maintenance (array of maintenance actions)
            Make sure contradictions with the primary model are clearly discussed.
            """
        ).strip()

    return system_prompt, user_prompt


__all__ = ["build_review_prompt"]
