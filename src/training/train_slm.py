"""
System 2: Agricultural SLM Fine-Tuning Pipeline (Ro-SLM Methodology)
====================================================================
Generates synthetic agricultural command + vision-to-dosing training data
and fine-tunes Qwen2.5-3B-Instruct (or 1.5B) using LoRA on the MI300X GPU.

Usage (on MI300X):
  python src/training/train_slm.py --generate-only
  python src/training/train_slm.py --model Qwen/Qwen2.5-3B-Instruct --epochs 3
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

COMMAND_TEMPLATES = [
    ("Apply {dose} mL of bio-fertilizer to the pomegranate base.", "fertilize"),
    ("Pour {dose}ml liquid nutrient at the pomegranate root zone.", "fertilize"),
    ("Dose {dose} mL NPK solution on this pomegranate sapling.", "fertilize"),
    ("Give {dose} mL organic fertilizer to the pomegranate tree.", "fertilize"),
    ("Anaar ke paudhe ko {dose} ml khad daalo.", "fertilize"),
    ("Irrigate pomegranate base with {dose} mL nutrient mix.", "fertilize"),
]


def generate_synthetic_dataset(output_path: Path, num_samples: int = 1200) -> Path:
    """Create balanced positive + safety-rejection instruction pairs."""
    random.seed(42)
    records: List[Dict] = []

    for i in range(num_samples):
        dose = random.choice([40, 50, 60, 80, 100, 120, 150, 200, 250, 300])
        tmpl, action = random.choice(COMMAND_TEMPLATES)
        cmd = tmpl.format(dose=dose)

        # 60% valid pomegranate, 20% non-pomegranate, 20% low-confidence
        case = random.random()
        if case < 0.60:
            is_pom = True
            conf = round(random.uniform(0.86, 0.99), 3)
            species = "Pomegranate (Punica granatum)"
            authorized = True
            reason = f"Verified pomegranate at {conf*100:.1f}% confidence; safe to dispense {dose} mL."
        elif case < 0.80:
            is_pom = False
            conf = round(random.uniform(0.20, 0.78), 3)
            species = random.choice(["Unverified Crop", "Weed / Grass Patch", "Bare Soil"])
            authorized = False
            reason = f"ABORT: Target is '{species}', not confirmed pomegranate."
        else:
            is_pom = True
            conf = round(random.uniform(0.55, 0.83), 3)
            species = "Pomegranate (Punica granatum)"
            authorized = False
            reason = f"ABORT: Confidence {conf:.2f} is below 0.85 safety threshold."

        vision_state = {
            "is_pomegranate": is_pom,
            "confidence": conf,
            "species": species,
            "weeds_at_base": random.choice([True, False]),
            "num_fruits": random.randint(1, 6) if is_pom else 0,
        }

        records.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are System 2, the agricultural safety supervisor for a pomegranate "
                        "dosing robot. Output strict JSON with keys: authorized, dose_volume_ml, reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Command: {cmd}\nVision: {json.dumps(vision_state)}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "authorized": authorized,
                        "dose_volume_ml": float(dose) if authorized else 0.0,
                        "reasoning": reason,
                    }),
                },
            ]
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[✓] Generated {len(records)} SLM training examples → {output_path}")
    return output_path


def train_lora(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
):
    """Fine-tune Qwen2.5 with LoRA using transformers + peft."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
    except ImportError as e:
        print(f"[✗] Missing package for SLM training: {e}")
        print("    Install on MI300X: pip install peft datasets")
        return

    print(f"[→] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw_ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(text, truncation=True, max_length=384)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized_ds = raw_ds.map(tokenize_fn, remove_columns=raw_ds.column_names)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=False,
        fp16=False,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()

    # Merge and save ready-to-serve model
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[✓] Merged SLM saved → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train System 2 Agricultural SLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="data/slm_agri_instruct.jsonl")
    parser.add_argument("--output", default="models/slm_lora")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    ds_path = generate_synthetic_dataset(Path(args.dataset), args.samples)
    if not args.generate_only:
        train_lora(
            args.model, ds_path, Path(args.output),
            epochs=args.epochs, batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
