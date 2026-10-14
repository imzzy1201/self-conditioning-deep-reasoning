"""OCR-based CER judge: extracts text from both target and predicted images via VLM, computes Character Error Rate."""

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from judge import extract_response_text, pil_image_to_data_url

OCR_PROMPT = (
    "You are an OCR system. Extract ALL text from this image. "
    "Output ONLY the extracted text characters, preserving every line break exactly as it appears in the image. "
    "This includes both paragraph breaks (explicit line breaks in the text) and word-wrap line breaks "
    "(where a line wraps to the next line due to width constraints). "
    "Each line in the image must correspond to a separate line in your output. "
    "Do NOT add any explanation, commentary, labels, or formatting markers. "
    "If the image contains no text, output nothing."
)


def compute_cer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Compute Character Error Rate between two strings.

    Returns (cer_value, edit_distance, ref_length).
    CER = edit_distance / max(len(reference), 1).
    """
    from rapidfuzz.distance import Levenshtein

    ref_len = len(reference)
    if ref_len == 0:
        if len(hypothesis) == 0:
            return (0.0, 0, 0)
        return (1.0, len(hypothesis), 0)

    edit_distance = Levenshtein.distance(reference, hypothesis)
    cer = edit_distance / ref_len
    return (cer, edit_distance, ref_len)


def ocr_image_with_vlm(
    client: Any,
    model: str,
    image: Image.Image,
    max_retries: int = 3,
) -> tuple[str | None, str, str, int, int]:
    """OCR a single PIL image via the VLM.

    Returns (extracted_text, raw_response, reasoning_content, prompt_tokens, completion_tokens).
    On failure, extracted_text is None and raw_response contains the error reason.
    """
    data_url = pil_image_to_data_url(image)
    content = [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                extra_body={"enable_thinking": True, "thinking_budget": 5000},
            )

            if len(response.choices) == 0:
                continue

            raw = extract_response_text(response.choices[0].message.content)
            reasoning = ""
            if getattr(response.choices[0].message, "reasoning_content", None) is not None:
                reasoning = extract_response_text(response.choices[0].message.reasoning_content)

            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage") and response.usage is not None:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)

            return (raw, raw, reasoning, prompt_tokens, completion_tokens)
        except Exception:  # noqa: BLE001
            continue

    return (None, f"OCR failed after {max_retries} attempts", "", 0, 0)


def ocr_cer_single_sample(
    client: Any,
    model: str,
    sample: dict[str, Any],
    global_step: int | None = None,
) -> dict[str, Any]:
    """OCR both target and predicted images, then compute CER.

    Returns a result dict with CER details and OCR outputs.
    """
    answer_image = sample.get("answer_image")
    pred_path = sample["pred_path"]

    # OCR target image
    gt_text, gt_raw, gt_reasoning, gt_pt, gt_ct = (None, "", "", 0, 0)
    if answer_image is not None:
        gt_text, gt_raw, gt_reasoning, gt_pt, gt_ct = ocr_image_with_vlm(client, model, answer_image)
    else:
        answer_path = sample.get("answer_path", "")
        if answer_path:
            try:
                with Image.open(answer_path) as img:
                    answer_image = img.convert("RGB")
                gt_text, gt_raw, gt_reasoning, gt_pt, gt_ct = ocr_image_with_vlm(client, model, answer_image)
            except Exception:  # noqa: BLE001
                gt_text = None
                gt_raw = f"Failed to load answer image: {answer_path}"

    # OCR predicted image
    pred_text, pred_raw, pred_reasoning, pred_pt, pred_ct = (None, "", "", 0, 0)
    try:
        with Image.open(pred_path) as img:
            pred_image = img.convert("RGB")
        pred_text, pred_raw, pred_reasoning, pred_pt, pred_ct = ocr_image_with_vlm(client, model, pred_image)
    except Exception:  # noqa: BLE001
        pred_text = None
        pred_raw = f"Failed to load predicted image: {pred_path}"

    # Compute CER
    if gt_text is not None and pred_text is not None:
        cer, edit_distance, ref_length = compute_cer(gt_text, pred_text)
        status = "valid"
    else:
        cer, edit_distance, ref_length = (None, None, None)
        status = "error"

    record: dict[str, Any] = {
        "index": sample.get("index", ""),
        "sample_name": sample.get("sample_name", ""),
        "pred_path": pred_path,
        "answer_path": str(sample.get("answer_path", "")),
        "ground_truth_source": "ocr_target",
        "gt_ocr_text": gt_text or "",
        "pred_ocr_text": pred_text or "",
        "cer": cer,
        "edit_distance": edit_distance,
        "ref_length": ref_length,
        "status": status,
        "gt_ocr_raw": gt_raw,
        "pred_ocr_raw": pred_raw,
        "gt_reasoning": gt_reasoning,
        "pred_reasoning": pred_reasoning,
        "prompt_tokens": gt_pt + pred_pt,
        "completion_tokens": gt_ct + pred_ct,
    }
    if global_step is not None:
        record["step"] = str(global_step)

    return record


def run_ocr_cer_judge(
    *,
    client: Any,
    model: str,
    samples: list[dict[str, Any]],
    global_step: int | None = None,
    max_workers: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run OCR + CER on all samples concurrently.

    Returns (results_list, token_usage_dict).
    """
    results: list[dict[str, Any] | None] = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, sample in enumerate(samples):
            futures[executor.submit(ocr_cer_single_sample, client, model, sample, global_step)] = idx
        for future in tqdm(as_completed(futures), total=len(samples), desc="OCR CER", unit="sample"):
            idx = futures[future]
            results[idx] = future.result()

    # Fill any missing results with error records
    final_results: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for idx, r in enumerate(results):
        if r is None:
            r = {
                "index": str(idx),
                "status": "error",
                "cer": None,
                "edit_distance": None,
                "ref_length": None,
                "gt_ocr_text": "",
                "pred_ocr_text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        total_prompt_tokens += r.get("prompt_tokens", 0)
        total_completion_tokens += r.get("completion_tokens", 0)
        final_results.append(r)

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
    return final_results, token_usage


def summarize_cer_results(
    results: list[dict[str, Any]],
    token_usage: dict[str, int] | None = None,
) -> dict[str, float]:
    """Aggregate per-sample CER into metrics dict."""
    valid = [r for r in results if r["status"] == "valid" and r["cer"] is not None]
    num_error = sum(1 for r in results if r["status"] == "error")
    cer_values = [r["cer"] for r in valid]
    perfect_count = sum(1 for c in cer_values if c == 0.0)

    metrics: dict[str, float] = {
        "eval/cer_num_valid": float(len(valid)),
        "eval/cer_num_error": float(num_error),
        "eval/cer_perfect_count": float(perfect_count),
    }
    if len(cer_values) > 0:
        metrics["eval/cer_mean"] = statistics.mean(cer_values)
        metrics["eval/cer_median"] = statistics.median(cer_values)

    if token_usage is not None:
        metrics["eval/total_prompt_tokens"] = float(token_usage.get("prompt_tokens", 0))
        metrics["eval/total_completion_tokens"] = float(token_usage.get("completion_tokens", 0))

    return metrics