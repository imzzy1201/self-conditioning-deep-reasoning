import csv
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class ColumnMulDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform_input: Optional[callable] = None,
        transform_output: Optional[callable] = None,
    ):
        self.root_dir = root_dir
        self.transform_input = transform_input
        self.transform_output = transform_output

        data_path = os.path.join(root_dir, "data.pt")
        metadata_path = os.path.join(root_dir, "metadata.csv")
        self.mode = "pt" if os.path.exists(data_path) else "csv"

        if self.mode == "pt":
            data = torch.load(data_path, weights_only=False)
            self.inputs = data["inputs"]
            self.outputs = data["outputs"]
            self.metadata = data["metadata"]
            self.config = data["config"]
        else:
            self.inputs = None
            self.outputs = None
            self.config = None
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.metadata = list(reader)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "pt":
            input_tensor = self.inputs[idx].float() / 255.0
            output_tensor = self.outputs[idx].float() / 255.0
            meta = self.metadata[idx]
        else:
            meta = self.metadata[idx]
            input_path = os.path.join(self.root_dir, meta["edit_image"])
            output_path = os.path.join(self.root_dir, meta["image"])
            input_image = Image.open(input_path).convert("RGB")
            output_image = Image.open(output_path).convert("RGB")
            input_tensor = pil_to_tensor(input_image).float() / 255.0
            output_tensor = pil_to_tensor(output_image).float() / 255.0

        if self.transform_input is not None:
            input_tensor = self.transform_input(input_tensor)
        if self.transform_output is not None:
            output_tensor = self.transform_output(output_tensor)

        return {
            "input": input_tensor,
            "output": output_tensor,
            "a": int(meta["a"]) if "a" in meta else None,
            "b": int(meta["b"]) if "b" in meta else None,
            "result": int(meta["result"]) if "result" in meta else None,
        }