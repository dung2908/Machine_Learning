import numpy as np

def grad(x):
    return 2 * x

def cost(x):
    return x**2 - 2

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x, it) = myGD1(x0=5.0, eta=0.1)

# 4. In kết quả
print("=== KẾT QUẢ BÀI 1 ===")
print(f"Giá trị x tìm được: {x[-1]:.4f}")
print(f"Giá trị f(x) tại cực tiểu: {cost(x[-1]):.4f}")
print(f"Số vòng lặp: {it}")