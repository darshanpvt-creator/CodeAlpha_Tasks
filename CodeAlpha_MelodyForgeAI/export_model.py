from pathlib import Path
import torch
import onnx
from train import MelodyLSTM, MODELS

model_path = MODELS / "melody_lstm.pth"
output_path = MODELS / "melody_lstm.onnx"

if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model = MelodyLSTM()
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

example_input = torch.tensor([[0, 2, 4, 7, 9, 7, 4, 2]], dtype=torch.long)

torch.onnx.export(
    model,
    (example_input,),
    str(output_path),
    input_names=["notes"],
    output_names=["logits"],
    dynamic_axes={
        "notes": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    dynamo=False
)

onnx_model = onnx.load(str(output_path))
onnx.checker.check_model(onnx_model)

print("ONNX export successful!")
print(f"Model saved: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
