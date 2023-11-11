import torch
import torchvision.models as models

# 构建模型并实例化
# model = models.vgg16(weights='IMAGENET1K_V1')
# torch.save(model.state_dict(), 'model_weights.pth') # 仅保存模型参数

# # 加载模型也需要先把模型实例化
# model = models.vgg16()
# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval() # 推理模式
# print(model.parameters())

# 保存模型参数和图结构
# model = models.vgg16(weights='IMAGENET1K_V1')
# torch.save(model, 'model.pth')
# model = torch.load('model.pth')
# print(model)

