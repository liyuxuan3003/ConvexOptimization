# Slide 07: Gradient Methods for Unconstrained Problems

本讲对应 `07_grad_descent_unconstrained.pdf`：无约束问题的梯度方法。知识点范围按 slide 走，逻辑上参考 Boyd Chapter 9，尤其是 `9.1`、`9.2`、`9.3` 中关于 descent methods、gradient descent、backtracking line search 和收敛分析的部分。

## 本讲主线

Slide 07 的主题是无约束优化：

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

其中 $f$ 是可微函数。整讲围绕一个问题展开：如何用梯度下降构造一列点 $x_0,x_1,x_2,\ldots$，使得 $f(x_t)$ 不断下降，并最终接近某种意义下的最优点或稳定点？

Slide 的结构大概是：

1. 一般下降算法和梯度下降。
2. 二次函数上的收敛分析。
3. 强凸且光滑函数上的线性收敛。
4. backtracking line search。
5. 没有强凸性时的收敛结果。
6. 非凸问题中只能追求小梯度和避开 saddle point。

## 1. 无约束可微优化

问题形式是：

$$
\min_x f(x), \qquad x \in \mathbb{R}^n
$$

假设 $f$ 可微。若 $f$ 是凸函数，那么全局最优点 $x^\star$ 满足一阶最优性条件：

$$
\nabla f(x^\star)=0
$$

这点和 Boyd Chapter 9 开头完全一致。Boyd 的说法是：无约束凸优化问题等价于解方程

$$
\nabla f(x)=0
$$

但一般无法解析求解，所以需要迭代算法。

迭代算法的基本形式是：

$$
x_{t+1}=x_t+\eta_t d_t
$$

其中：

- $d_t$ 是搜索方向。
- $\eta_t>0$ 是步长。
- 希望 $f(x_{t+1})<f(x_t)$。

## 2. 下降方向

Slide 定义了方向导数：

$$
f'(x;d)
=
\lim_{\tau \downarrow 0}
\frac{f(x+\tau d)-f(x)}{\tau}
=
\nabla f(x)^\top d
$$

如果

$$
\nabla f(x)^\top d < 0
$$

那么 $d$ 就是 $x$ 处的下降方向。

直观理解：沿着方向 $d$ 走一小步，函数值会下降。

所以一个一般下降算法是：

$$
x_{t+1}=x_t+\eta_t d_t
$$

只要 $d_t$ 是下降方向，且 $\eta_t$ 选得足够小，就能保证函数值下降。

## 3. 梯度下降

梯度下降选择最自然的下降方向：

$$
d_t=-\nabla f(x_t)
$$

所以更新公式是：

$$
x_{t+1}
=
x_t-\eta_t \nabla f(x_t)
$$

为什么负梯度是自然的？因为对于所有单位方向 $\|d\|_2\le 1$，方向导数为

$$
\nabla f(x)^\top d
$$

由 Cauchy-Schwarz 不等式，

$$
\nabla f(x)^\top d
\ge
-\|\nabla f(x)\|_2\|d\|_2
\ge
-\|\nabla f(x)\|_2
$$

等号在

$$
d=-\frac{\nabla f(x)}{\|\nabla f(x)\|_2}
$$

时取得。

因此，负梯度方向是欧氏范数意义下让函数值下降最快的方向。所以梯度下降也叫 steepest descent。

## 4. 先看二次函数

Slide 先从二次函数开始，是因为二次函数能把梯度下降的行为看得非常清楚。

考虑：

$$
f(x)
=
\frac{1}{2}(x-x^\star)^\top Q(x-x^\star)
$$

其中 $Q \succ 0$。这时

$$
\nabla f(x)
=
Q(x-x^\star)
$$

梯度下降为：

$$
x_{t+1}
=
x_t-\eta Q(x_t-x^\star)
$$

两边减去 $x^\star$：

$$
x_{t+1}-x^\star
=
(I-\eta Q)(x_t-x^\star)
$$

递推得到：

$$
x_t-x^\star
=
(I-\eta Q)^t(x_0-x^\star)
$$

所以收敛速度由矩阵 $I-\eta Q$ 的谱半径决定。

设 $Q$ 的最大特征值为 $\lambda_1(Q)$，最小特征值为 $\lambda_n(Q)$。如果选择

$$
\eta
=
\frac{2}{\lambda_1(Q)+\lambda_n(Q)}
$$

则有：

$$
\|x_t-x^\star\|_2
\le
\left(
\frac{\lambda_1(Q)-\lambda_n(Q)}
{\lambda_1(Q)+\lambda_n(Q)}
\right)^t
\|x_0-x^\star\|_2
$$

定义条件数：

$$
\kappa
=
\frac{\lambda_1(Q)}{\lambda_n(Q)}
$$

则收敛因子可以写成：

$$
\frac{\kappa-1}{\kappa+1}
$$

这说明：

- 若 $\kappa$ 接近 $1$，收敛很快。
- 若 $\kappa$ 很大，收敛很慢。
- 二次函数的等高线越狭长，梯度下降越容易 zig-zag。

这和 Boyd Chapter 9 中对 condition number 的解释一致：梯度下降对函数等高线的几何形状很敏感。

## 5. 线性收敛是什么意思

Slide 说这个叫 linear convergence 或 geometric convergence。

如果误差满足：

$$
e_t \le c^t e_0,
\qquad 0<c<1
$$

就叫线性收敛。

注意这里的 "linear" 不是指 $e_t$ 像直线一样下降，而是指在半对数图上：

$$
\log e_t
\le
t\log c+\log e_0
$$

是一条直线。

所以 linear convergence 其实是几何级数收敛。

## 6. Exact Line Search

前面最优固定步长需要知道 $Q$ 的谱信息：

$$
\lambda_1(Q),\lambda_n(Q)
$$

这在实际问题中通常不知道。因此 slide 引入 exact line search：

$$
\eta_t
=
\arg\min_{\eta\ge 0}
f(x_t-\eta \nabla f(x_t))
$$

也就是每一步都沿负梯度方向找最优步长。

对二次函数，令

$$
g_t=\nabla f(x_t)
$$

则 exact line search 的步长为：

$$
\eta_t
=
\frac{g_t^\top g_t}{g_t^\top Qg_t}
$$

slide 给出的结论是：

$$
f(x_t)-f(x^\star)
\le
\left(
\frac{\lambda_1(Q)-\lambda_n(Q)}
{\lambda_1(Q)+\lambda_n(Q)}
\right)^{2t}
\left(f(x_0)-f(x^\star)\right)
$$

这仍然是线性收敛，收敛速度仍然由条件数控制。

这里有一个容易误解的点：exact line search 每一步都选该方向上的最优步长，但它不一定比最优固定步长在理论阶数上更快。因为问题的根本瓶颈是负梯度方向本身，而不是只在这个方向上选步长。

## 7. 强凸且光滑函数

二次函数之后，slide 推广到一般函数。

假设 $f$ 是二阶可微的，并且满足：

$$
\mu I \preceq \nabla^2 f(x) \preceq L I,
\qquad \forall x
$$

这表示：

- $f$ 是 $\mu$-strongly convex。
- $f$ 是 $L$-smooth。
- 条件数为：

$$
\kappa=\frac{L}{\mu}
$$

这正是二次函数情况的推广。对于二次函数，$\mu=\lambda_n(Q)$，$L=\lambda_1(Q)$。

强凸给下界：

$$
f(y)
\ge
f(x)+\nabla f(x)^\top (y-x)
+
\frac{\mu}{2}\|y-x\|_2^2
$$

光滑给上界：

$$
f(y)
\le
f(x)+\nabla f(x)^\top (y-x)
+
\frac{L}{2}\|y-x\|_2^2
$$

直观上：

- 强凸说明函数至少像曲率为 $\mu$ 的二次函数一样弯。
- 光滑说明函数至多像曲率为 $L$ 的二次函数一样陡。
- $\kappa=L/\mu$ 衡量函数的"狭长程度"。

## 8. 强凸光滑下的梯度下降收敛

Slide 的 Theorem 2.1 是：

若 $f$ 是 $\mu$-strongly convex 且 $L$-smooth，选择

$$
\eta
=
\frac{2}{\mu+L}
$$

则

$$
\|x_t-x^\star\|_2
\le
\left(
\frac{\kappa-1}{\kappa+1}
\right)^t
\|x_0-x^\star\|_2
$$

其中

$$
\kappa=\frac{L}{\mu}
$$

这和二次函数完全同型。

证明思路是利用 Hessian 的积分表示。因为

$$
\nabla f(x_t)-\nabla f(x^\star)
=
\left(
\int_0^1 \nabla^2 f(x_\tau)\,d\tau
\right)(x_t-x^\star)
$$

其中

$$
x_\tau=x_t+\tau(x^\star-x_t)
$$

又因为 $\nabla f(x^\star)=0$，所以：

$$
\nabla f(x_t)
=
H_t(x_t-x^\star)
$$

其中

$$
H_t=
\int_0^1 \nabla^2 f(x_\tau)\,d\tau
$$

并且

$$
\mu I \preceq H_t \preceq L I
$$

于是：

$$
x_{t+1}-x^\star
=
x_t-x^\star-\eta \nabla f(x_t)
=
(I-\eta H_t)(x_t-x^\star)
$$

选择 $\eta=2/(\mu+L)$ 后得到 contraction：

$$
\|x_{t+1}-x^\star\|_2
\le
\frac{L-\mu}{L+\mu}
\|x_t-x^\star\|_2
$$

也就是：

$$
\|x_{t+1}-x^\star\|_2
\le
\frac{\kappa-1}{\kappa+1}
\|x_t-x^\star\|_2
$$

这就是 slide 的结论。

## 9. 目标函数值收敛

由 $L$-smooth 可知：

$$
f(x_t)-f(x^\star)
\le
\frac{L}{2}\|x_t-x^\star\|_2^2
$$

所以距离误差的线性收敛可以推出函数值误差的线性收敛：

$$
f(x_t)-f(x^\star)
\le
\frac{L}{2}
\left(
\frac{\kappa-1}{\kappa+1}
\right)^{2t}
\|x_0-x^\star\|_2^2
$$

这就是 slide 里 "a direct consequence using smoothness" 的含义。

## 10. Backtracking Line Search

实际中我们通常不知道 $\mu$ 和 $L$，也不想固定步长。因此 slide 介绍 backtracking line search。

算法是标准的 Armijo backtracking：

给定 $0<\alpha<1/2$，$0<\beta<1$，从 $\eta=1$ 开始，如果不满足：

$$
f(x_t-\eta \nabla f(x_t))
\le
f(x_t)-\alpha \eta \|\nabla f(x_t)\|_2^2
$$

就令：

$$
\eta \leftarrow \beta \eta
$$

直到满足为止。

这个条件叫 Armijo condition。它要求函数值下降至少达到一个可接受的量：

$$
\alpha \eta \|\nabla f(x_t)\|_2^2
$$

直观上，backtracking 的逻辑是：

1. 先大胆试 $\eta=1$。
2. 如果下降不够，就缩小步长。
3. 因为 $f$ 光滑，步长足够小时一定会满足 Armijo 条件。

Boyd Chapter 9.2 也是这样组织的：先定义 general descent method，再定义 backtracking line search，用它保证 sufficient decrease。

## 11. Backtracking 下的线性收敛

Slide 的 Theorem 2.2 来自 Boyd：

若 $f$ 是 $\mu$-strongly convex 且 $L$-smooth，使用 backtracking line search，则：

$$
f(x_t)-f(x^\star)
\le
\left(
1-\min\left\{2\mu\alpha,\frac{2\beta\alpha\mu}{L}\right\}
\right)^t
\left(f(x_0)-f(x^\star)\right)
$$

这仍然是线性收敛。

它说明：

- 不知道 $L$ 也没关系。
- backtracking 会自动找到足够小但不太小的步长。
- 收敛因子仍然受 $\mu/L$ 控制。

Boyd 的解释是：backtracking 本质上在估计局部 Lipschitz 常数。

## 12. 强凸性是否必要

Slide 接下来问：强凸性是线性收敛的必要条件吗？

答案：不是。

强凸性可以被放松，例如：

- local strong convexity
- regularity condition
- Polyak-Lojasiewicz condition

这些条件都比全局强凸弱，但仍然可以保证某种线性收敛。

## 13. Logistic Regression 的例子

Slide 用 logistic regression 说明全局强凸可能不成立。

目标函数大致为：

$$
f(x)
=
\frac{1}{m}
\sum_{i=1}^m
\log\left(1+\exp(-y_i a_i^\top x)\right)
$$

它的 Hessian 中包含类似权重：

$$
\frac{\exp(-y_i a_i^\top x)}
{\left(1+\exp(-y_i a_i^\top x)\right)^2}
a_i a_i^\top
$$

当 $\|x\|\to\infty$ 时，这些权重可能趋近于 $0$，所以 Hessian 的最小特征值可能趋近于 $0$。因此它不是全局强凸的。

但是在初始点附近的一个局部球内，Hessian 可能有正下界：

$$
\mu I \preceq \nabla^2 f(x) \preceq L I,
\qquad x \in B_0
$$

其中

$$
B_0=
\{x:\|x-x^\star\|_2\le \|x_0-x^\star\|_2\}
$$

只要迭代点一直留在这个球里，前面的强凸光滑分析仍然成立。

这就是 local strong convexity 的思想。

## 14. Regularity Condition

Slide 还给了一个 regularity condition：

$$
\langle \nabla f(x),x-x^\star\rangle
\ge
\frac{\mu}{2}\|x-x^\star\|_2^2
+
\frac{1}{2L}\|\nabla f(x)\|_2^2
$$

这个条件只比较 $x$ 和某个最优点 $x^\star$，不像强凸性那样要求任意两点 $x,y$ 都满足曲率下界。

若取步长：

$$
\eta=\frac{1}{L}
$$

则可以证明：

$$
\|x_t-x^\star\|_2^2
\le
\left(1-\frac{\mu}{L}\right)^t
\|x_0-x^\star\|_2^2
$$

证明直接展开：

$$
x_{t+1}
=
x_t-\frac{1}{L}\nabla f(x_t)
$$

所以：

$$
\|x_{t+1}-x^\star\|_2^2
=
\left\|x_t-x^\star-\frac{1}{L}\nabla f(x_t)\right\|_2^2
$$

展开后用 regularity condition 即可得到 contraction。

## 15. PL 条件

Polyak-Lojasiewicz condition 是：

$$
\|\nabla f(x)\|_2^2
\ge
2\mu\left(f(x)-f(x^\star)\right)
$$

它的含义是：只要函数值还没到最优，梯度就不能太小。

这个条件有两个重要特点：

- 它保证每个 stationary point 都是 global minimizer。
- 它不要求 $f$ 是强凸函数。
- 它甚至不必保证全局最优点唯一。

若 $f$ 是 $L$-smooth 且满足 PL 条件，取

$$
\eta=\frac{1}{L}
$$

则：

$$
f(x_t)-f(x^\star)
\le
\left(1-\frac{\mu}{L}\right)^t
\left(f(x_0)-f(x^\star)\right)
$$

这说明 PL 条件足以保证目标函数值线性收敛。

## 16. 过参数线性回归

Slide 用 over-parameterized linear regression 说明 PL 条件的意义。

问题是：

$$
\min_{x\in\mathbb{R}^n}
f(x)
=
\frac{1}{2}
\sum_{i=1}^m
(a_i^\top x-y_i)^2
$$

写成矩阵形式：

$$
f(x)
=
\frac{1}{2}\|Ax-y\|_2^2
$$

如果 $n>m$，也就是参数维度大于样本数，则

$$
\nabla^2 f(x)=A^\top A
$$

秩最多为 $m$，所以 $A^\top A$ 奇异，函数不是强凸的。

但是如果 $A\in\mathbb{R}^{m\times n}$ 满行秩，则 $AA^\top$ 正定。梯度为：

$$
\nabla f(x)=A^\top(Ax-y)
$$

于是：

$$
\|\nabla f(x)\|_2^2
=
(Ax-y)^\top AA^\top(Ax-y)
\ge
\lambda_{\min}(AA^\top)\|Ax-y\|_2^2
$$

而

$$
f(x)=\frac{1}{2}\|Ax-y\|_2^2
$$

所以：

$$
\|\nabla f(x)\|_2^2
\ge
2\lambda_{\min}(AA^\top)f(x)
$$

这就是 PL 条件。

因此，即使问题不是强凸的，梯度下降仍然可以在线性回归的目标函数值上达到线性收敛：

$$
f(x_t)-f(x^\star)
\le
\left(
1-\frac{\lambda_{\min}(AA^\top)}
{\lambda_{\max}(AA^\top)}
\right)^t
\left(f(x_0)-f(x^\star)\right)
$$

这里还有一个 slide 提到的现代机器学习观点：虽然全局最优解不唯一，GD 会倾向于收敛到离初始点 $x_0$ 最近的全局最优解，这叫 implicit bias。

## 17. 只假设凸且光滑

接下来 slide 去掉强凸性，只保留：

- $f$ convex
- $f$ $L$-smooth

此时不一定有唯一最优点，也不一定有线性收敛。

这时更自然的目标不是控制距离：

$$
\|x_t-x^\star\|_2
$$

而是控制目标函数误差：

$$
f(x_t)-f(x^\star)
$$

例如 slide 提到 $f(x)=1/x$，$x>0$。最优点在无穷远处，$x^\star=\infty$ 不是普通意义下的点，但函数值可以趋近最优值 $0$。

所以没有强凸性时，关注 objective improvement 更合理。

## 18. Smoothness 给出的下降保证

由 $L$-smooth，有：

$$
f(y)
\le
f(x)+\nabla f(x)^\top(y-x)
+
\frac{L}{2}\|y-x\|_2^2
$$

令

$$
y=x_t-\eta \nabla f(x_t)
$$

则：

$$
f(x_{t+1})-f(x_t)
\le
-\eta\|\nabla f(x_t)\|_2^2
+
\frac{L\eta^2}{2}\|\nabla f(x_t)\|_2^2
$$

也就是：

$$
f(x_{t+1})-f(x_t)
\le
-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla f(x_t)\|_2^2
$$

取

$$
\eta=\frac{1}{L}
$$

得到：

$$
f(x_{t+1})
\le
f(x_t)
-
\frac{1}{2L}
\|\nabla f(x_t)\|_2^2
$$

这就是 slide 的 Fact 2.7。

注意它不依赖凸性，只依赖光滑性。所以对非凸函数也成立。

## 19. 凸光滑下的距离单调性

Slide 的 Fact 2.8 说：

若 $f$ convex 且 $L$-smooth，取 $\eta=1/L$，则对任意最优点 $x^\star$，

$$
\|x_{t+1}-x^\star\|_2^2
\le
\|x_t-x^\star\|_2^2
-
\frac{1}{L^2}
\|\nabla f(x_t)\|_2^2
$$

特别地：

$$
\|x_{t+1}-x^\star\|_2
\le
\|x_t-x^\star\|_2
$$

也就是说，GD 的迭代点不会离某个最优点越来越远。

这个性质用到了 convex + smooth 的一个重要不等式，也叫 cocoercivity：

$$
\langle \nabla f(x)-\nabla f(y),x-y\rangle
\ge
\frac{1}{L}
\|\nabla f(x)-\nabla f(y)\|_2^2
$$

令 $y=x^\star$，且 $\nabla f(x^\star)=0$，即可得到距离下降。

## 20. 凸光滑下的 $O(1/t)$ 收敛

Slide 的 Theorem 2.9 是：

若 $f$ convex 且 $L$-smooth，取 $\eta=1/L$，则：

$$
f(x_t)-f(x^\star)
\le
\frac{2L\|x_0-x^\star\|_2^2}{t}
$$

所以达到

$$
f(x_t)-f(x^\star)\le \epsilon
$$

需要：

$$
O\left(\frac{1}{\epsilon}\right)
$$

次迭代。

这比线性收敛慢很多。对比：

- 强凸光滑：$O(\log(1/\epsilon))$
- 凸光滑：$O(1/\epsilon)$

为什么变慢？因为没有强凸性时，函数可能越来越平，梯度变小不一定意味着离最优点非常近。

## 21. 非凸问题

最后 slide 转向非凸优化。

非凸问题中可能存在：

- local minima
- saddle points
- bad landscape
- 多个 stationary points

一般来说，不可能希望一个算法高效找到全局最优点。

因此非凸优化中常见目标变成：

$$
\|\nabla f(x)\|_2 \le \epsilon
$$

这样的点叫 $\epsilon$-approximate stationary point。

## 22. 非凸光滑下的小梯度保证

即使 $f$ 非凸，只要 $f$ 是 $L$-smooth，且使用

$$
\eta=\frac{1}{L}
$$

仍有下降不等式：

$$
f(x_{k+1})
\le
f(x_k)
-
\frac{1}{2L}
\|\nabla f(x_k)\|_2^2
$$

把 $k=0,\ldots,t-1$ 求和，得到 telescoping sum：

$$
\frac{1}{2L}
\sum_{k=0}^{t-1}
\|\nabla f(x_k)\|_2^2
\le
f(x_0)-f(x_t)
\le
f(x_0)-f(x^\star)
$$

因此：

$$
\min_{0\le k<t}
\|\nabla f(x_k)\|_2
\le
\sqrt{
\frac{2L(f(x_0)-f(x^\star))}{t}
}
$$

所以要找到一个 $\epsilon$-approximate stationary point，需要：

$$
O\left(\frac{1}{\epsilon^2}\right)
$$

次迭代。

注意这个结论只说明轨迹里存在某个点梯度小，不说明最后一个点一定收敛到 stationary point。

## 23. Saddle Point

Slide 最后讨论 saddle point。

有些 stationary point 是局部极小点，有些是 saddle point。梯度下降如果刚好初始化在 saddle point 上，会被困住，因为：

$$
\nabla f(x_0)=0
$$

则：

$$
x_{t+1}=x_t
$$

但是如果随机初始化，在很多温和条件下，梯度下降几乎必然不会收敛到严格 saddle point。

Slide 举了非凸二次函数：

$$
f(x)=\frac{1}{2}x^\top A x
$$

其中

$$
A=u_1u_1^\top-u_2u_2^\top
$$

这里 $x=0$ 是 saddle point。GD 更新为：

$$
x_{t+1}=(I-\eta A)x_t
$$

由于 $A$ 在 $u_2$ 方向有负曲率，对应 $I-\eta A$ 在该方向的特征值为 $1+\eta>1$，所以只要随机初始化在 $u_2$ 方向上有非零分量，该分量会被放大，迭代会离开 saddle。

这解释了 slide 的结论：随机初始化时，GD 几乎不会卡在 saddle point。

## 本讲核心总结

这讲的逻辑可以压缩成一条线：

$$
\text{GD}
\rightarrow
\text{quadratic case}
\rightarrow
\text{strongly convex and smooth}
\rightarrow
\text{convex and smooth}
\rightarrow
\text{nonconvex smooth}
$$

对应结论是：

| 假设 | 步长 | 主要结论 |
|---|---:|---|
| 二次函数 $Q\succ 0$ | $2/(\lambda_1+\lambda_n)$ | 距离线性收敛 |
| $\mu$-strongly convex + $L$-smooth | $2/(\mu+L)$ | 距离线性收敛 |
| $\mu$-strongly convex + $L$-smooth + backtracking | 自适应 | 函数值线性收敛 |
| PL + $L$-smooth | $1/L$ | 函数值线性收敛 |
| convex + $L$-smooth | $1/L$ | 函数值 $O(1/t)$ 收敛 |
| nonconvex + $L$-smooth | $1/L$ | $O(1/\epsilon^2)$ 找到小梯度点 |

## 和 Boyd 的关系

Boyd Chapter 9 更经典，主要讲：

- descent methods
- backtracking line search
- gradient descent under strong convexity
- steepest descent
- Newton method

Slide 07 比 Boyd 更偏现代一阶优化理论，额外强调：

- PL condition
- over-parameterized linear regression
- nonconvex stationary point
- saddle escaping

所以理解这讲时，可以用 Boyd 帮你建立"下降法、光滑性、强凸性、backtracking"的清晰基础，再用 slide 补充现代机器学习里常见的弱条件和非凸情形。
