import torch
from ultralytics import YOLO

def export_model_to_onnx(pt_path="/home/bahrombek/Desktop/RailSafeAI/models/best.pt",
                         onnx_path="best.onnx",
                         opset=12,
                         imgsz=1024,
                         simplify=True,
                         dynamic=True):
    # Modelni yuklash
    model = YOLO(pt_path)

    # Eksport qilish
    model.export(
        format="onnx",          # ONNX format
        opset=opset,            # ONNX opset versiyasi
        imgsz=imgsz,            # Treningda ishlatilgan image size
        simplify=simplify,      # ONNXni soddalashtirish
        dynamic=dynamic         # Dinamik shape (turli o'lchamdagi inputlarga moslashadi)
    )

    print(f"\n✅ Model '{pt_path}' dan ONNX '{onnx_path}' ga eksport qilindi.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    export_model_to_onnx()
