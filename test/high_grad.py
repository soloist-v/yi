import yi

a = yi.tensor([1, 2, 3, 4, 5, 6.], requires_grad=True)
y = a * a * a * 2 + 3 * a * a + a
print("y", y)
grad1 = yi.grad(y, a, retain_graph=True)
print("grad1", grad1)
grad2 = yi.grad(grad1, a, retain_graph=True)
print("grad2", grad2)
grad3 = yi.grad(grad2, a, retain_graph=True)
print("grad3", grad3)
grad4 = yi.grad(grad3, a, retain_graph=True)
print("grad4", grad4)
