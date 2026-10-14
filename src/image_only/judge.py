import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm.auto import tqdm


JUDGE_THINKING_BUDGET = 1000  # Max tokens for judge's reasoning content


def sanitize_name(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = stem.replace(" ", "_")
    # Strip characters invalid in Windows filenames (e.g. ':' from shard URLs).
    for ch in "\\/:*?\"<>|":
        stem = stem.replace(ch, "_")
    return stem


def pil_image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_judge_output(raw: str) -> tuple[bool | None, str]:
    text = raw.strip()
    if text == "":
        return None, ""

    lowered = text.lower()
    if "[yes]" in lowered or lowered == "yes" or lowered == "[yes]":
        return True, text
    if "[no]" in lowered or lowered == "no" or lowered == "[no]":
        return False, text
    if "yes" in lowered and "no" not in lowered:
        return True, text
    if "no" in lowered and "yes" not in lowered:
        return False, text

    return None, text


def get_openai_client_and_model() -> tuple[Any, str]:
    env_api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL")

    if env_api_key is None or env_api_key.strip() == "":
        raise ValueError("Missing OPENAI_API_KEY in environment/.env")
    if base_url is None or base_url.strip() == "":
        raise ValueError("Missing OPENAI_API_BASE in environment/.env")
    if model is None or model.strip() == "":
        raise ValueError("Missing OPENAI_MODEL in environment/.env")

    client = OpenAI(api_key=env_api_key.strip(), base_url=base_url.strip())
    return client, model.strip()


def load_eval_prompt_template(path_str: str) -> str:
    prompt_path = Path(path_str)
    if not prompt_path.is_absolute():
        prompt_path = (Path.cwd() / prompt_path).resolve()

    if not prompt_path.exists():
        return ""

    text = prompt_path.read_text(encoding="utf-8").strip()
    return text


def build_judge_content(
    eval_prompt: str,
    gt_data_url: str,
    pred_data_url: str,
    edit_data_url: str | None = None,
    prompt_text: str = "",
) -> list[dict[str, Any]]:
    prompt_token = "{{prompt_text}}"
    edit_token = "{{edit_image}}"
    solution_token = "{{solution_image}}"
    predicted_token = "{{predicted_image}}"

    if solution_token not in eval_prompt or predicted_token not in eval_prompt:
        raise ValueError(
            "eval_prompt must contain both placeholders: {{solution_image}} and {{predicted_image}}"
        )

    chunks = re.split(
        r"(\{\{prompt_text\}\}|\{\{edit_image\}\}|\{\{solution_image\}\}|\{\{predicted_image\}\})",
        eval_prompt,
    )
    content: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk == "":
            continue
        if chunk == prompt_token:
            content.append({"type": "text", "text": prompt_text if prompt_text.strip() else "(No text prompt provided)"})
        elif chunk == edit_token:
            if edit_data_url is None:
                content.append({"type": "text", "text": "(No condition image provided)"})
            else:
                content.append({"type": "image_url", "image_url": {"url": edit_data_url}})
        elif chunk == solution_token:
            content.append({"type": "image_url", "image_url": {"url": gt_data_url}})
        elif chunk == predicted_token:
            content.append({"type": "image_url", "image_url": {"url": pred_data_url}})
        else:
            content.append({"type": "text", "text": chunk})
    return content


def extract_response_text(response_content: Any) -> str:
    if isinstance(response_content, str):
        return response_content
    if isinstance(response_content, list):
        parts: list[str] = []
        for part in response_content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(response_content)


def judge_with_openai(
    client: Any,
    model: str,
    eval_prompt: str,
    answer_image: Image.Image,
    pred_image: Image.Image,
    edit_image: Image.Image | None = None,
    prompt_text: str = "",
) -> tuple[bool | None, str, str, str, int, int]:
    max_retries = 3
    last_reason = ""
    last_raw = ""

    for _ in range(max_retries):
        try:
            edit_data_url = pil_image_to_data_url(edit_image) if edit_image is not None else None
            gt_data_url = pil_image_to_data_url(answer_image)
            pred_data_url = pil_image_to_data_url(pred_image)
            content = build_judge_content(
                eval_prompt=eval_prompt,
                gt_data_url=gt_data_url,
                pred_data_url=pred_data_url,
                edit_data_url=edit_data_url,
                prompt_text=prompt_text,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                extra_body={
                    "enable_thinking": True,
                    "thinking_budget": JUDGE_THINKING_BUDGET,
                },
            )

            if len(response.choices) == 0:
                last_reason = "Empty response choices from OpenAI API"
                last_raw = ""
                continue

            raw = extract_response_text(response.choices[0].message.content)
            reasoning = extract_response_text(response.choices[0].message.reasoning_content) if getattr(response.choices[0].message, "reasoning_content", None) else ""

            # Extract token usage from response
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage") and response.usage is not None:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)

            correct, reason = parse_judge_output(raw)
            if correct is None:
                last_reason = "Failed to parse judge output as YES/NO"
                last_raw = raw
                continue

            return correct, reason, raw, reasoning, prompt_tokens, completion_tokens
        except Exception as e:  # noqa: BLE001
            last_reason = f"{type(e).__name__}: {e}"
            last_raw = ""

    if last_reason == "":
        return None, f"OpenAI judge failed after {max_retries} attempts", last_raw, "", 0, 0
    return None, f"OpenAI judge failed after {max_retries} attempts: {last_reason}", last_raw, "", 0, 0


def judge_single_sample(
    client: Any,
    model: str,
    eval_prompt: str,
    answer_path: str,
    pred_path: str,
    global_step: int | None,
    answer_image: Image.Image | None = None,
    edit_image: Image.Image | None = None,
    prompt_text: str = "",
) -> dict[str, str]:
    """Judge one sample.

    If PIL images are provided, use them directly; otherwise load answer/prediction
    from disk. Keeping the condition image as a PIL object avoids shard URL/path
    ambiguity and preserves the exact input used for generation.
    """
    if answer_image is None:
        with Image.open(answer_path) as img:
            answer_image = img.convert("RGB")
    else:
        answer_image = answer_image.copy()
    if edit_image is not None:
        edit_image = edit_image.copy()
    with Image.open(pred_path) as img:
        pred_image = img.convert("RGB")

    correct, judge_reason, judge_raw, reasoning, prompt_tokens, completion_tokens = judge_with_openai(
        client=client,
        model=model,
        eval_prompt=eval_prompt,
        answer_image=answer_image,
        pred_image=pred_image,
        edit_image=edit_image,
        prompt_text=prompt_text,
    )

    if correct is None:
        judge_label = "error"
    else:
        judge_label = "true" if correct else "false"

    record = {
        "pred_path": pred_path,
        "answer_path": answer_path,
        "correct": judge_label,
        "reason": judge_reason,
        "raw": judge_raw,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if reasoning:
        record["reasoning_content"] = reasoning
    if global_step is not None:
        record["step"] = str(global_step)

    return record


def _aggregate_votes(votes: list[dict[str, str]], sample: dict[str, Any], k: int, strict: bool, global_step: int | None) -> dict[str, str]:
    """Collect k votes for one sample and produce the final judge record."""
    correct_vals = [v["correct"] for v in votes]
    yes_count = correct_vals.count("true")
    no_count = correct_vals.count("false")

    if strict:
        if len(set(correct_vals)) > 1 or yes_count + no_count < k:
            judge_label = "error"
        else:
            judge_label = correct_vals[0]
    else:
        judge_label = "true" if yes_count > no_count else "false"

    record = {
        "pred_path": sample["pred_path"],
        "answer_path": sample["answer_path"],
        "correct": judge_label,
        "judge_details": votes,
    }
    if global_step is not None:
        record["step"] = str(global_step)
    return record


def run_judge_in_parallel(
    *,
    client: Any,
    model: str,
    eval_prompt: str,
    samples: list[dict[str, Any]],
    global_step: int | None = None,
    max_workers: int = 50,
    k: int = 2,
    strict: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Run judge on all samples * k votes concurrently, then aggregate per sample.

    Returns:
        Tuple of (judge_results, token_usage) where token_usage contains
        total prompt_tokens and completion_tokens across all API calls.
    """
    total = len(samples) * k
    raw_results: list[dict | None] = [None] * total

    def _call_one(task_idx: int, sample_idx: int) -> tuple[int, dict]:
        sample = samples[sample_idx]
        record = judge_single_sample(
            client=client,
            model=model,
            eval_prompt=eval_prompt,
            answer_path=sample["answer_path"],
            pred_path=sample["pred_path"],
            global_step=global_step,
            answer_image=sample.get("answer_image"),
            edit_image=sample.get("edit_image_pil"),
            prompt_text=sample.get("prompt", ""),
        )
        return task_idx, record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for si, sample in enumerate(samples):
            for vi in range(k):
                task_idx = si * k + vi
                futures[executor.submit(_call_one, task_idx, si)] = task_idx
        for future in tqdm(as_completed(futures), total=total, desc="Judge", unit="vote"):
            task_idx, result = future.result()
            raw_results[task_idx] = result

    # Aggregate votes per sample and sum token usage
    judge_results: list[dict[str, str]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for si, sample in enumerate(samples):
        start = si * k
        votes = [raw_results[start + vi] for vi in range(k) if raw_results[start + vi] is not None]
        if len(votes) < k:
            # Fill missing with error
            while len(votes) < k:
                votes.append({"correct": "error", "reason": "API call failed", "raw": "", "reasoning_content": "", "prompt_tokens": 0, "completion_tokens": 0})
        # Sum token usage from all votes for this sample
        for vote in votes:
            total_prompt_tokens += vote.get("prompt_tokens", 0)
            total_completion_tokens += vote.get("completion_tokens", 0)
        record = _aggregate_votes(votes, sample, k, strict, global_step)
        record["index"] = sample.get("index", str(si))
        record["sample_name"] = sample.get("sample_name", "")
        record["image"] = sample.get("image", "")
        record["edit_image"] = sample.get("edit_image", "")
        judge_results.append(record)

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
    return judge_results, token_usage


def summarize_judge_results(judge_results: list[dict[str, str]], token_usage: dict[str, int] | None = None) -> dict[str, float]:
    valid = [item for item in judge_results if item["correct"] in {"true", "false"}]
    num_correct = sum(1 for item in valid if item["correct"] == "true")
    num_error = sum(1 for item in judge_results if item["correct"] == "error")
    num_unknown = sum(1 for item in judge_results if item["correct"] == "unknown")

    metrics: dict[str, float] = {
        "eval/judge_num_valid": float(len(valid)),
        "eval/judge_num_correct": float(num_correct),
        "eval/judge_num_error": float(num_error),
        "eval/judge_num_unknown": float(num_unknown),
    }
    if len(valid) > 0:
        metrics["eval/judge_accuracy"] = num_correct / len(valid)

    if token_usage is not None:
        metrics["eval/judge_prompt_tokens"] = float(token_usage.get("prompt_tokens", 0))
        metrics["eval/judge_completion_tokens"] = float(token_usage.get("completion_tokens", 0))

    return metrics


def write_jsonl(path: str, records: list[dict[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
