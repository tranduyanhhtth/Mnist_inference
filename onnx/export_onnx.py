import os
import sys
import torch
# import onnx
# import onnx.version_converter as version_converter
# Thêm đường dẫn cho module export_pth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))
from export_pth import Trainer, BetterModel, DataLoader

class ONNXExporter:
    def __init__(self, model):
        self.model = model

    def load_model(self, path='mnist_model.pth'):
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            sys.exit(1)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()
        print(f"Model loaded from {path}")

    def export_to_onnx(self, dummy_input, export_path="mnist_model.onnx"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        dummy_input = dummy_input.to(device)

        torch.onnx.export(self.model, dummy_input, export_path,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"Model exported to {export_path}")

    # def convert_opset(self, model_path, target_opset=11, output_path="mnist_model_opset11.onnx"):
    #     # Đọc và chuyển đổi mô hình sang target_opset
    #     model = onnx.load(model_path)
    #     converted_model = version_converter.convert_version(model, target_opset)
    #     onnx.save(converted_model, output_path)
    #     print(f"Model converted to opset {target_opset} and saved to {output_path}")


# -------- Main Execution -------- #
if __name__ == "__main__":
    # Data loading
    data_loader = DataLoader(batch_size=32)
    train_loader, val_loader = data_loader.load_data()

    # Initialize model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterModel()
    trainer = Trainer(model, train_loader, val_loader, device)

    # Training
    trainer.train()

    # Save model
    trainer.save_model()

    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    exporter = ONNXExporter(model)
    exporter.load_model()
    exporter.export_to_onnx(dummy_input)
    # exporter.convert_opset("mnist_model.onnx", target_opset=11, output_path="mnist_model_opset11.onnx")