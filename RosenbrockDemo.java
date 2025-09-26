// RosenbrockDemo.java
// Single-file demo: Rosenbrock function + BFGS optimizer with Armijo backtracking.
// No external deps. Tested on Java 11+.

public class RosenbrockDemo {

    // Rosenbrock parameters (classics: a=1, b=100)
    static final double A = 1.0;
    static final double B = 100.0;

    // Evaluate f(x,y) = (a-x)^2 + b(y - x^2)^2
    static double f(double[] x) {
        double xv = x[0], yv = x[1];
        return (A - xv) * (A - xv) + B * (yv - xv * xv) * (yv - xv * xv);
    }

    // Gradient of Rosenbrock
    // df/dx = -2(a - x) - 4b x (y - x^2)
    // df/dy =  2b (y - x^2)
    static void grad(double[] x, double[] g) {
        double xv = x[0], yv = x[1];
        g[0] = -2.0 * (A - xv) - 4.0 * B * xv * (yv - xv * xv);
        g[1] =  2.0 * B * (yv - xv * xv);
    }

    // Dot product
    static double dot(double[] a, double[] b) {
        return a[0]*b[0] + a[1]*b[1];
    }

    // 2x2 matrix-vector multiply
    static void matVec(double[][] M, double[] v, double[] out) {
        out[0] = M[0][0]*v[0] + M[0][1]*v[1];
        out[1] = M[1][0]*v[0] + M[1][1]*v[1];
    }

    // BFGS optimizer for 2D with Armijo backtracking
    static Result bfgs(double[] x0, int maxIter, double tol) {
        double[] x = new double[]{x0[0], x0[1]};
        double[] g = new double[2];
        grad(x, g);

        // Initial inverse Hessian as identity
        double[][] H = new double[][]{{1.0, 0.0}, {0.0, 1.0}};

        int iter = 0;
        double fx = f(x);
        System.out.printf("%-6s %-14s %-14s %-14s %-14s%n", "iter", "x", "y", "f(x,y)", "||grad||");
        while (iter < maxIter) {
            double gn = Math.sqrt(dot(g, g));
            System.out.printf("%-6d %-14.6f %-14.6f %-14.6e %-14.6e%n",
                    iter, x[0], x[1], fx, gn);

            if (gn < tol) break;

            // Search direction: p = -H * g
            double[] p = new double[2];
            double[] ng = new double[]{-g[0], -g[1]};
            matVec(H, ng, p);

            // Armijo backtracking line search
            double alpha = lineSearchArmijo(x, fx, g, p);

            // Update x
            double[] xNew = new double[]{x[0] + alpha * p[0], x[1] + alpha * p[1]};
            double fxNew = f(xNew);

            // y = g_new - g
            double[] gNew = new double[2];
            grad(xNew, gNew);
            double[] s = new double[]{xNew[0] - x[0], xNew[1] - x[1]};
            double[] yv = new double[]{gNew[0] - g[0], gNew[1] - g[1]};

            // BFGS update: H_{k+1} = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            double sy = s[0]*yv[0] + s[1]*yv[1];
            if (Math.abs(sy) > 1e-12) {
                double rho = 1.0 / sy;

                // Compute (I - rho * s y^T)
                double[][] I = new double[][]{{1,0},{0,1}};
                double[][] A = new double[2][2];
                // A = I - rho * s y^T
                A[0][0] = 1.0 - rho * s[0] * yv[0];
                A[0][1] =      - rho * s[0] * yv[1];
                A[1][0] =      - rho * s[1] * yv[0];
                A[1][1] = 1.0 - rho * s[1] * yv[1];

                // B = I - rho * y s^T
                double[][] B = new double[2][2];
                B[0][0] = 1.0 - rho * yv[0] * s[0];
                B[0][1] =      - rho * yv[0] * s[1];
                B[1][0] =      - rho * yv[1] * s[0];
                B[1][1] = 1.0 - rho * yv[1] * s[1];

                // temp = A * H
                double[][] temp = mul2x2(A, H);
                // H = temp * B
                H = mul2x2(temp, B);

                // Add rho * s s^T
                H[0][0] += rho * s[0] * s[0];
                H[0][1] += rho * s[0] * s[1];
                H[1][0] += rho * s[1] * s[0];
                H[1][1] += rho * s[1] * s[1];
            } else {
                // fall back to identity if curvature is bad
                H[0][0] = 1; H[0][1] = 0;
                H[1][0] = 0; H[1][1] = 1;
            }

            // Prepare for next iter
            x = xNew;
            fx = fxNew;
            g = gNew;
            iter++;
        }

        double gn = Math.sqrt(dot(g, g));
        System.out.printf("%-6d %-14.6f %-14.6f %-14.6e %-14.6e%n",
                iter, x[0], x[1], fx, gn);

        return new Result(x, fx, iter, gn);
    }

    // 2x2 matrix multiplication
    static double[][] mul2x2(double[][] M, double[][] N) {
        double[][] R = new double[2][2];
        R[0][0] = M[0][0]*N[0][0] + M[0][1]*N[1][0];
        R[0][1] = M[0][0]*N[0][1] + M[0][1]*N[1][1];
        R[1][0] = M[1][0]*N[0][0] + M[1][1]*N[1][0];
        R[1][1] = M[1][0]*N[0][1] + M[1][1]*N[1][1];
        return R;
    }

    // Armijo backtracking line search
    static double lineSearchArmijo(double[] x, double fx, double[] g, double[] p) {
        double c = 1e-4;      // Armijo parameter
        double tau = 0.5;     // step shrink
        double alpha = 1.0;   // initial step
        double gp = dot(g, p);

        // Ensure descent direction (should be for BFGS, but just in case)
        if (gp >= 0) {
            p[0] = -g[0];
            p[1] = -g[1];
            gp = -dot(g, g);
        }

        // Backtrack until sufficient decrease
        while (true) {
            double fxTry = f(new double[]{x[0] + alpha * p[0], x[1] + alpha * p[1]});
            if (fxTry <= fx + c * alpha * gp) break;
            alpha *= tau;
            if (alpha < 1e-12) break;
        }
        return alpha;
    }

    // Convenience container
    static class Result {
        final double[] x;
        final double fval;
        final int iters;
        final double gradNorm;
        Result(double[] x, double fval, int iters, double gradNorm) {
            this.x = x; this.fval = fval; this.iters = iters; this.gradNorm = gradNorm;
        }
    }

    public static void main(String[] args) {
        double x0 = -1.2;
        double y0 =  1.0;
        int    maxIter = 1000;
        double tol = 1e-8;

        try {
            if (args.length >= 1) x0 = Double.parseDouble(args[0]);
            if (args.length >= 2) y0 = Double.parseDouble(args[1]);
            if (args.length >= 3) maxIter = Integer.parseInt(args[2]);
            if (args.length >= 4) tol = Double.parseDouble(args[3]);
        } catch (Exception e) {
            System.out.println("Usage: java RosenbrockDemo [x0 y0 maxIters tol]");
            System.out.println("Falling back to defaults.");
        }

        double[] start = new double[]{x0, y0};
        Result r = bfgs(start, maxIter, tol);
        System.out.println();
        System.out.printf("Final: x=%.10f, y=%.10f%n", r.x[0], r.x[1]);
        System.out.printf("f(x,y)=%.12e  |grad|=%.3e  iters=%d%n", r.fval, r.gradNorm, r.iters);
        System.out.println("Expected optimum near (1, 1) with f=0.");
    }
}
