"""
MI300X System 2 SLM Fine-Tuning & Evaluation (Pomegranate Agronomy + Decision Brain)
====================================================================================
Fine-tunes Qwen2.5-3B-Instruct using bf16 LoRA (r=32, all linear layers) with
assistant-only loss masking, merges the adapter, and evaluates JSON validity,
safety veto recall, and pomegranate agronomy Q&A on a held-out split.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def tokenize_with_assistant_mask(example: Dict[str, Any], tokenizer, max_length: int = 768) -> Dict[str, Any]:
    msgs = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_tok = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full_tok["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": full_tok["attention_mask"],
        "labels": labels,
    }


def evaluate_model(model, tokenizer, eval_path: Path, report_path: Path) -> Dict[str, Any]:
    model.eval()
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    dec_total = 0
    dec_json_ok = 0
    unsafe_total = 0
    unsafe_vetoed = 0
    safe_total = 0
    safe_correct = 0

    intent_total = 0
    intent_json_ok = 0
    intent_conf_ok = 0

    qa_samples: List[Dict[str, str]] = []
    latencies_ms: List[float] = []

    for row in rows:
        cat = row.get("category", "")
        msgs = row["messages"]
        prompt_msgs = msgs[:-1]
        expected_str = msgs[-1]["content"]

        inp_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(inp_text, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        pred = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if cat == "decision_v2":
            dec_total += 1
            exp_json = json.loads(expected_str)
            exp_auth = bool(exp_json["authorized"])
            try:
                pred_json = json.loads(pred)
                dec_json_ok += 1
                pred_auth = bool(pred_json.get("authorized"))
            except Exception:
                pred_json = None
                pred_auth = False  # fail-closed parse

            if not exp_auth:
                unsafe_total += 1
                if not pred_auth:
                    unsafe_vetoed += 1
            else:
                safe_total += 1
                if pred_json is not None and pred_auth:
                    safe_correct += 1

        elif cat == "intent_parse":
            intent_total += 1
            try:
                pred_json = json.loads(pred)
                intent_json_ok += 1
                if float(pred_json.get("min_confidence", 0.0)) >= 0.80:
                    intent_conf_ok += 1
            except Exception:
                pass
        else:
            if len(qa_samples) < 10:
                qa_samples.append({
                    "question": prompt_msgs[-1]["content"],
                    "expected": expected_str,
                    "predicted": pred,
                })

    lat_sorted = sorted(latencies_ms)
    p95_ms = lat_sorted[int(0.95 * len(lat_sorted))] if lat_sorted else 0.0
    summary = {
        "decision_total": dec_total,
        "decision_json_valid_pct": round(100.0 * dec_json_ok / max(1, dec_total), 2),
        "unsafe_total": unsafe_total,
        "safety_veto_recall_pct": round(100.0 * unsafe_vetoed / max(1, unsafe_total), 2),
        "safe_total": safe_total,
        "safe_auth_accuracy_pct": round(100.0 * safe_correct / max(1, safe_total), 2),
        "intent_total": intent_total,
        "intent_json_valid_pct": round(100.0 * intent_json_ok / max(1, intent_total), 2),
        "intent_conf_floor_pct": round(100.0 * intent_conf_ok / max(1, intent_total), 2),
        "latency_p95_ms": round(p95_ms, 1),
        "qa_samples": qa_samples,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[EVAL SUMMARY]", json.dumps({k: v for k, v in summary.items() if k != "qa_samples"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", default="data/slm_pomegranate_train.jsonl")
    parser.add_argument("--eval-file", default="data/slm_pomegranate_eval.jsonl")
    parser.add_argument("--output-dir", default="/scratch/runs/slm_pomegranate_3b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.10,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[2/4] Tokenizing train & validation datasets...")
    raw_ds = load_dataset("json", data_files=args.train_file, split="train")
    tok_ds = raw_ds.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, args.max_length),
        remove_columns=raw_ds.column_names,
    )
    eval_raw = load_dataset("json", data_files=args.eval_file, split="train")
    eval_tok = eval_raw.map(
        lambda ex: tokenize_with_assistant_mask(ex, tokenizer, args.max_length),
        remove_columns=eval_raw.column_names,
    )

    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        warmup_steps=40,
        max_grad_norm=1.0,
        neftune_noise_alpha=5.0,
        bf16=True,
        fp16=False,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    print("[3/4] Starting NEFTune + Eval-Loss Early-Stop LoRA training on MI300X...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tok_ds,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    train_res = trainer.train()
    print("[TRAIN METRICS]", train_res.metrics)

    merged_dir = out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[OK] Merged model saved -> {merged_dir}")

    print(f"[4/4] Evaluating on held-out split: {args.eval_file}")
    evaluate_model(merged, tokenizer, Path(args.eval_file), out_dir / "eval_report.json")


if __name__ == "__main__":
    main()
