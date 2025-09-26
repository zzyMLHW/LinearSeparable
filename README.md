# 线性可分性判断算法

本项目实现了基于感知机学习算法的线性可分性判断程序，能够自动判断给定的数据集是否线性可分，并可视化结果。

## 算法原理

### 线性可分性定义

对于二分类问题，如果存在一个超平面能够完美地将两类数据点分开，则称该数据集是**线性可分**的。

在二维空间中，线性可分意味着存在一条直线 $L: \mathbf{w}^\mathrm{T} \mathbf{x} + b = 0$，使得：
- 对于标签为 $+1$ 的点：$\mathbf{w}^\mathrm{T} \mathbf{x} + b > 0$
- 对于标签为 $-1$ 的点：$\mathbf{w}^\mathrm{T} \mathbf{x} + b < 0$

### 感知机学习算法

感知机算法是解决线性可分问题的经典算法，其核心思想是：

1. **初始化**：权重向量 $\mathbf{w}_0 = \mathbf{0}$，偏置 $b_0 = 0$

2. **迭代更新**：对于第 $k$ 个epoch，遍历所有样本 $(\mathbf{x}_i, y_i)$：
   - 如果 $y_i(\mathbf{w}_k^\mathrm{T} \mathbf{x}_i + b_k) \leq 0$（误分类）
   - 则更新：$\mathbf{w}_{k+1} \leftarrow \mathbf{w}_k + y_i \mathbf{x}_i$，$b_{k+1} \leftarrow b_k + y_i$

3. **收敛条件**：当某个epoch中所有点都被正确分类时，算法收敛

### 增广向量表示

为了简化计算，我们将偏置 $b$ 纳入权重向量中：

$$\mathbf{x}' = \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix}, \quad \mathbf{w}' = \begin{bmatrix} \mathbf{w} \\ b \end{bmatrix}$$

此时决策函数变为：$f(\mathbf{x}') = \mathbf{w}'^\mathrm{T} \mathbf{x}'$。

## 快速开始

### 环境要求

- Python 3.7+
- numpy >= 1.21.0
- matplotlib >= 3.5.0

### 安装依赖

```bash
pip install -r requirements.txt
```


### 运行程序

```bash
python main.py
```

### 预期输出

程序运行后会输出：

```
对于第一个数据集 X1, 输出为 Y = 1 (预期: 1)
找到的分割线权重: w1=-4.000, w2=7.000, b=-2.000
分割线方程: -4.000*x + 7.000*y + -2.000 = 0
对于第二个数据集 X2, 输出为 Y = -1 (预期: -1)
对于三维数据集 X3, 输出为 Y = 1 (预期: 1)
对于三维数据集 X4, 输出为 Y = -1 (预期: -1)
```

同时会在 `results/` 目录下生成可视化图像：
- `X1.png`：线性可分数据集（包含分割线）
- `X2.png`：线性不可分数据集

### 项目结构

```
LinearSeparable/
├── main.py              # 主程序文件
├── requirements.txt     # 依赖包列表
├── README.md           # 项目说明文档
├── results/            # 输出结果目录
│   ├── X1.png         # X1数据集可视化
│   └── X2.png         # X2数据集可视化
└── .gitignore         # Git忽略文件
```