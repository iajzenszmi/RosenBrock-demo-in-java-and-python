# rosenbrock.py                  import numpy as np               import matplotlib.pyplot as plt  import pandas as pd
                                 # Rosenbrock function and gradient
def rosenbrock(xy, a=1.0, b=100.0):              x, y = xy   
return (a - x)**2 + b*(y - x**2)**2  
                                                                             def rosenbrock_grad(xy, a=1.0, b=100.0):
    x, y = xy                        dfdx = -2*(a - x) - 4*b*x*(y - x**2)                              dfdy = 2*b*(y - x**2)
    return np.array([dfdx, dfdy])                                 # Backtracking line search       def backtracking_line_search(f, grad, x, p, alpha0=1.0, rho=0.5, c=1e-4):                              fx = f(x)
    g = grad(x)                      alpha = alpha0                   while f(x + alpha * p) > fx + c * alpha * np.dot(g, p):
        alpha *= rho                     if alpha < 1e-12:                    break                    return alpha
                                 # Gradient descent               def gradient_descent(f, grad, x0, max_iter=5000, tol=1e-8):
    x = np.array(x0, dtype=float)    history = []                     for k in range(max_iter):            g = grad(x)
        gnorm = np.linalg.norm(g)        fx = f(x)                        history.append((k, x[0], x[1], fx, gnorm))
        if gnorm < tol:                      break                        p = -g                           alpha = backtracking_line_search(f, grad, x, p)                   x = x + alpha * p            return np.array(history), x  
def main():                          # Start point                    x0 = [-1.2, 1.0]                 hist, xstar = gradient_descent(rosenbrock, rosenbrock_grad, x0)                                
    # Save iterations to CSV         df = pd.DataFrame(hist, columns=["iter", "x", "y", "f(x,y)", "||grad||"])
    df.to_csv("rosenbrock_iterations.csv", index=False)                                                # Contour plot
    xlin = np.linspace(-2, 2, 400)                                    ylin = np.linspace(-1, 3, 400)                                    X, Y = np.meshgrid(xlin, ylin)                                    Z = (1 - X)**2 + 100*(Y - X**2)**2                                                                 plt.figure()                     cs = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))          plt.plot(df["x"], df["y"], marker="o", linewidth=1)               plt.scatter([1], [1], marker="x", c="red")                        plt.title("Rosenbrock Contour + Path")                            plt.xlabel("x"); plt.ylabel("y")                                  plt.savefig("rosenbrock_contour_path.png", dpi=160)
                                     # Convergence plot               plt.figure()                     plt.semilogy(df["iter"], df["f(x,y)"])                            plt.title("Convergence on Rosenbrock")                            plt.xlabel("Iteration"); plt.ylabel("f(x,y)")                     plt.grid(True, linestyle=":")    plt.savefig("rosenbrock_convergence.png", dpi=160)
                                     print(f"Estimated minimizer: ({xstar[0]:.8f}, {xstar[1]:.8f})")
    print("CSV saved as rosenbrock_iterations.csv")                   print("Plots saved as rosenbrock_contour_path.png and rosenbrock_convergence.png")                                              if __name__ == "__main__":
    main()                       userland@localhost:~$