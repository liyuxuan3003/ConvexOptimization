# Slide 08: Gradient Methods for Constrained Problems

本讲对应 `08_grad_descent_constrained.pdf`：有约束问题的梯度方法。知识点范围按 slide 走，逻辑上参考 Boyd 教材相关内容。

## 本讲主线

Slide 08 的主题是约束优化：

$$
\min_{x} f(x) \qquad \text{s.t.}\ x\in C
$$

其中 $f$ 是凸函数，$C\subseteq\mathbb{R}^n$ 是闭凸集。Slide 08 介绍两大类处理约束的一阶方法：Frank-Wolfe 算法和投影梯度下降。

Slide 的结构大概是：

1. 可行方向方法的一般框架。
2. Frank-Wolfe 算法及其收敛性。
3. 投影梯度下降及其收敛性，分强凸光滑和凸光滑两种情况讨论。

## 1. 约束凸问题

问题形式：

$$
\min_x f(x), \qquad x\in C
$$

其中：
- $f$ 是凸函数。
- $C$ 是闭凸集。

和有约束优化一样，关键难点是迭代点必须始终保持在 $C$ 内。

## 2. 可行方向方法

一般框架是：

$$
x_{t+1}=x_t+\eta_t d_t
$$

其中 $d_t$ 必须是可行方向，即 $x_t+\eta_t d_t\in C$。

核心问题：如何在保证可行性的同时，确保函数值下降？

两类主流方法给出了不同的答案：

- Frank-Wolfe：在 $C$ 内做线性优化，每次更新是凸组合，自动保持可行。
- 投影梯度下降：先做一步无约束梯度下降，再投影回 $C$。

---

## 3. Frank-Wolfe 算法

### 算法描述

Frank-Wolfe 也叫 conditional gradient method。每步做两件事：

1. 在 $C$ 上极小化一个线性函数：
   $$
   y_t=\arg\min_{x\in C}\langle\nabla f(x_t), x\rangle
   $$
   这一步叫 direction finding。

2. 凸组合更新：
   $$
   x_{t+1}=(1-\eta_t)x_t+\eta_t y_t
   $$
   因为 $x_t,y_t\in C$ 且 $C$ 是凸集，$x_{t+1}$ 自动留在 $C$ 内。

### 直观理解

把目标函数在当前点做一阶 Taylor 展开：

$$
f(x)\approx f(x_t)+\langle\nabla f(x_t),x-x_t\rangle
$$

极小化这个线性近似等价于极小化 $\langle\nabla f(x_t),x\rangle$。所以第一步是在 $C$ 上求这个线性近似的最小点 $y_t$，然后沿 $y_t$ 方向走一步。

因为每步只需要解一个**线性优化**子问题（在 $C$ 上极小化一个线性函数），所以当 $C$ 上的线性优化很便宜时，Frank-Wolfe 特别有吸引力。例如 $C$ 是多面体时，线性优化就是 LP，有成熟高效的解法。

### 步长选择

slide 给了两种选择：
- Line search：在 $[0,1]$ 上精确搜索。
- 预定步长：
  $$
  \eta_t=\frac{2}{t+2}
  $$
  这个步长随迭代递减，越往后更新越保守、越偏向保留当前点 $x_t$。

### 收敛性（凸光滑）

**Theorem 3.1**：若 $f$ 凸且 $L$-smooth，$C$ 有界，取 $\eta_t=\frac{2}{t+2}$，则

$$
f(x_t)-f(x^\star)\le\frac{2L d_C^2}{t+2}
$$

其中 $d_C=\sup_{x,y\in C}\|x-y\|_2$ 是可行集的最大直径。

所以 Frank-Wolfe 达到 $\epsilon$-accuracy 需要 $O(1/\epsilon)$ 次迭代，和无约束的凸光滑 GD 同阶。

证明从 smoothness 出发：

$$
\begin{aligned}
f(x_{t+1})-f(x_t)
&\le\langle\nabla f(x_t),x_{t+1}-x_t\rangle+\frac{L}{2}\|x_{t+1}-x_t\|_2^2 \\
&=\eta_t\langle\nabla f(x_t),y_t-x_t\rangle+\frac{L\eta_t^2}{2}\|y_t-x_t\|_2^2 \\
&\le\eta_t\langle\nabla f(x_t),x^\star-x_t\rangle+\frac{L\eta_t^2}{2}d_C^2 \\
&\le\eta_t\big(f(x^\star)-f(x_t)\big)+\frac{L\eta_t^2}{2}d_C^2
\end{aligned}
$$

第一行是 smoothness，第二行用更新式代入，第三行用 $y_t$ 是 linear minimization 的解（所以 $\langle\nabla f(x_t),y_t\rangle\le\langle\nabla f(x_t),x^\star\rangle$），第四行用凸性。令 $\Delta_t=f(x_t)-f(x^\star)$ 得到递推：

$$
\Delta_{t+1}\le(1-\eta_t)\Delta_t+\frac{L d_C^2}{2}\eta_t^2
$$

代入 $\eta_t=\frac{2}{t+2}$ 后用归纳法即得最终界。

### 强凸性不能改善 Frank-Wolfe 的收敛阶

对无约束 GD，强凸可以把 $O(1/t)$ 提升为线性收敛。但对 Frank-Wolfe，**一般不成立**。

slide 给了一个反例（Canon & Cullum, 68）：即使 $f$ 强凸，FW 也只能达到 $O(1/t)$ 阶，无法改进。直观原因是：当最优点在可行集边界上且不是极端点时，FW 的搜索方向只能在极端点里选，永远无法直接指向最优点。

改进需要额外假设，比如可行集 $C$ 本身是强凸的（如 $\ell_2$ 球）。

### Frank-Wolfe 的额外特点

- 优点：不需要投影，只需解线性优化。当投影很贵但线性优化便宜时特别合适。
- 缺点：收敛阶不能通过强凸性改善；每步是在 $C$ 内部做凸组合，所以最终解永远是 extreme points 的凸组合，天然带稀疏性。
- 可以用于非凸问题：slide 举了一个非凸二次的例子，FW 退化成了 power method。

---

## 4. 投影梯度下降

### 算法描述

先做一步无约束梯度更新，再投影到 $C$：

$$
x_{t+1}=P_C\big(x_t-\eta_t\nabla f(x_t)\big)
$$

其中 $P_C$ 是欧氏投影：

$$
P_C(x)=\arg\min_{z\in C}\|x-z\|_2^2
$$

这是一类最标准的有约束一阶方法。适合投影 $P_C$ 计算高效的场合。

### 投影定理

若 $C$ 是闭凸集，$x_C=P_C(x)$ 是 $x$ 在 $C$ 上的投影，当且仅当

$$
(x-x_C)^\top(z-x_C)\le 0,\qquad\forall z\in C
$$

几何含义：从 $x$ 指向 $x_C$ 的向量与从 $x_C$ 指向 $C$ 内任意点的向量夹角 $\ge 90^\circ$。

由此可得下降方向性质。令 $y_t=x_t-\eta_t\nabla f(x_t)$ 为无约束更新点，$x_{t+1}=P_C(y_t)$。投影定理给出：

$$
\nabla f(x_t)^\top(x_{t+1}-x_t)\ge 0
$$

换句话说，更新方向 $x_{t+1}-x_t$ 与负梯度之间的夹角不超过 $90^\circ$，所以它仍然是一个下降方向。

### 非扩张性

投影算子满足

$$
\|P_C(x)-P_C(z)\|_2\le\|x-z\|_2
$$

即投影不增加距离。这个性质保证了投影不会破坏无约束 GD 的收敛性。

### 强凸光滑 + 最优点在内部

若 $x^\star\in\operatorname{int}(C)$（所以 $\nabla f(x^\star)=0$），取 $\eta_t=\frac{2}{\mu+L}$，则收敛性和无约束完全一样：

$$
\|x_t-x^\star\|_2\le\left(\frac{\kappa-1}{\kappa+1}\right)^t\|x_0-x^\star\|_2,\qquad\kappa=\frac{L}{\mu}
$$

证明很简单：无约束 GD 已经给出了 $\|y_t-x^\star\|_2\le\frac{\kappa-1}{\kappa+1}\|x_t-x^\star\|_2$，投影的非扩张性保证了 $\|P_C(y_t)-P_C(x^\star)\|_2\le\|y_t-x^\star\|_2$，而 $x^\star\in C$ 所以 $P_C(x^\star)=x^\star$。因此有约束的误差被无约束的误差控制住。

### 强凸光滑（一般情况）

若不保证 $x^\star$ 在内部，取 $\eta_t=\frac{1}{L}$ 仍然有线性收敛，但收敛因子略弱：

$$
\|x_t-x^\star\|_2^2\le\left(1-\frac{\mu}{L}\right)^t\|x_0-x^\star\|_2^2
$$

证明的核心是引入广义梯度 $g_C(x)=L(x-P_C(x-\frac{1}{L}\nabla f(x)))$，它推广了无约束情况下的 $\nabla f(x)$，且满足一条类似 regularity condition 的不等式：

$$
\langle g_C(x),x-x^\star\rangle\ge\frac{\mu}{2}\|x-x^\star\|_2^2+\frac{1}{2L}\|g_C(x)\|_2^2
$$

代入 GD 分析框架即得。

### 凸光滑

若 $f$ 只是凸且 $L$-smooth，取 $\eta=\frac{1}{L}$：

$$
f(x_t)-f(x^\star)\le\frac{3L\|x_0-x^\star\|_2^2+f(x_0)-f(x^\star)}{t+1}
$$

仍然是 $O(1/t)$ 收敛，和无约束凸光滑 GD 同阶。

证明也是引入 $g_C(x)$，建立广义 smoothness 不等式（Lemma 3.9）：

$$
f(y)\ge f(x^+)+g_C(x)^\top(y-x)+\frac{1}{2L}\|g_C(x)\|_2^2
$$

然后照搬无约束的三步证明框架：smoothness 给下降 → 凸性给梯度下界 → 递推得 $O(1/t)$。

---

## 本讲核心总结

| 方法 | 假设 | 收敛阶 | 迭代复杂度 |
|---|---|---|---|
| Frank-Wolfe | convex + $L$-smooth | $O(1/t)$ | $O(1/\epsilon)$ |
| Frank-Wolfe | 加 strong convexity | 仍 $O(1/t)$（不能改善） | — |
| 投影 GD | $\mu$-strongly convex + $L$-smooth（$x^\star$ 在内部） | 线性，因子 $\frac{\kappa-1}{\kappa+1}$ | $O(\kappa\log(1/\epsilon))$ |
| 投影 GD | $\mu$-strongly convex + $L$-smooth（一般） | 线性，因子 $1-\mu/L$ | $O(\kappa\log(1/\epsilon))$ |
| 投影 GD | convex + $L$-smooth | $O(1/t)$ | $O(1/\epsilon)$ |

**Frank-Wolfe vs 投影 GD**

| | Frank-Wolfe | 投影 GD |
|---|---|---|
| 每步计算 | 解线性优化子问题 | 计算投影 $P_C$ |
| 强凸改善 | 不能 | 能 |
| 适合场景 | 投影贵但线性优化便宜 | 投影便宜 |
| 解的结构 | 极端点凸组合，天然稀疏 | 无特别结构 |

**和无约束 GD 的关系**

有约束 GD 的收敛分析本质上是在无约束 GD 框架上做两步替换：把 $\nabla f(x)$ 替换为广义梯度 $g_C(x)$，把 smoothness / strong convexity 的一阶不等式替换为对应的广义版本。证明结构完全平行，只是每一步都要处理投影引入的几何细节。
