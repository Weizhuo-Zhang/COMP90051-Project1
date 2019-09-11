from bert_serving.client import BertClient
import torch, numpy
from models.coconut_model_v13 import CoconutModel

bc = BertClient()
bc_encode = bc.encode(['First do it'])


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

checkpoint_filename = 'checkpoints/v13_best_epoch1.pth'
checkpoints = torch.load(checkpoint_filename, map_location=device)
model = CoconutModel()
model.load_state_dict(checkpoints["model_state_dict"])
model.eval()
model.to(device)
feature_list = None
pred = model(['First do it'], is_pair=False, feature_type='Mean')
pred = pred.detach().numpy()


dist = numpy.linalg.norm(pred - bc_encode)
print(dist)
# print(pred)
# print(bc_encode)
