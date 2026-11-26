import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("models/slm_lora")
mdl = AutoModelForCausalLM.from_pretrained(
    "models/slm_lora", torch_dtype=torch.bfloat16, device_map="auto"
)

cases = [
    ("POS", {"is_pomegranate": True, "confidence": 0.94, "species": "Pomegranate", "weeds_at_base": True, "num_fruits": 4}),
    ("NEG", {"is_pomegranate": False, "confidence": 0.42, "species": "Weed / Grass Patch", "weeds_at_base": True, "num_fruits": 0}),
]

for label, vis in cases:
    msgs = [
        {"role": "system", "content": "You are System 2, the agricultural safety supervisor for a pomegranate dosing robot. Output strict JSON with keys: authorized, dose_volume_ml, reasoning."},
        {"role": "user", "content": f"Command: Apply 80 mL of bio-fertilizer to the pomegranate base.\nVision: {json.dumps(vis)}"},
    ]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    out = mdl.generate(inp, max_new_tokens=100, do_sample=False)
    print(f"{label} -> {tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)}")
