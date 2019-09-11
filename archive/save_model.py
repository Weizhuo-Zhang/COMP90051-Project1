import torch
import os
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from models.coconut_model_v13 import CoconutModel
output_dir = "checkpoints/v13"

coconut_model = CoconutModel()
filename = 'checkpoints/v13_best.pth'
checkpoints = torch.load(filename, map_location='cpu')
coconut_model.load_state_dict(checkpoints["model_state_dict"])

model = coconut_model.bert_model

# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
coconut_model.tokenizer.save_vocabulary(output_dir)
