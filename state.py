from models.SearchCifarResNet_criteria import SearchWidthCifarResNet
import torch
import os

model = SearchWidthCifarResNet('ResNetBasicblock', 56, 10)

paths = os.path.join(os.getcwd(), 'result', 'state_dict', 'search', 'checkpoint.th')
checkpoint = torch.load(paths)

a = checkpoint['search_model']
model.load_state_dict(a)
#print(model.state_dict()['width_attention'][:40])
#print(model.state_dict()['criteria_attention'][:5])
print(model.state_dict()['layers.1.conv_a.weights'][0][0][0])
#print(model.state_dict()['layers.1.conv_a.BNs.weight'][:2])
#b = model.state_dict()['layers.1.conv_a.BNs.weight']
#print(model.state_dict()['layers.1.conv_a.BNs.running_mean'][:2])
#print(model.state_dict()['layers.1.conv_a.BNs.running_mean'])
#c = model.state_dict()['layers.1.conv_a.BNs.running_mean']
#print(b/c)
d = model.state_dict()['width_attention'][:5]
e = model.state_dict()['criteria_attention'][:5]
print(d)
print(e)
