import torch
x = torch.tensor([1., 2., 3.])
print(x, x.shape, x.dtype)

A = torch.randn(2, 3)
B = torch.randn(3, 4)
print((A @ B).shape)          # 矩阵乘
print(A.reshape(3, 2).shape)  # reshape
print(A.sum(), A.mean())

# 自动求导核心例子
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x**2 + 3*x
z = y.sum()
z.backward()
print(x.grad)   # 应为 2x+3 = [5., 7., 9.]