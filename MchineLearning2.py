import numpy as np


def grad(x):
    return x**2 - 1

def cost(x):
    return (1/3) * (x**3) - x


def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x, it) = myGD1(x0=0.5, eta=0.1)

print("=== KẾT QUẢ BÀI 2 ===")
print(f"Giá trị x tìm được: {x[-1]:.4f}")
print(f"Giá trị g(x) tại cực tiểu: {cost(x[-1]):.4f}")
print(f"Số vòng lặp: {it}")