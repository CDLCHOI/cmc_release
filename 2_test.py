import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

encoder = FrozenEncoder()

# 冻结 encoder 的参数
for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()

# 模拟 diffusion 输出
predict_x0 = torch.randn(1, 4, requires_grad=True)
x0 = torch.randn(1, 4)

# 提取感知特征
with torch.no_grad():
    feat_x0 = encoder(x0)

feat_predict = encoder(predict_x0)
loss = F.mse_loss(feat_predict, feat_x0)

loss.backward()

print(predict_x0.grad)  # ✅ 非 None，说明梯度成功回传到了 predict_x0
