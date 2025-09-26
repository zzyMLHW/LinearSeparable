import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union


def whetherLinearSeparable_numpy(X: np.ndarray, return_weights: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
    """
    判断一个点集是否是线性可分的。

    该函数使用感知机学习算法（Perceptron Algorithm）的原理。
    如果数据集是线性可分的，感知机算法保证在有限次迭代内收敛。
    我们会运行该算法一个固定的最多次数：
    - 如果在此期间算法收敛（即找到一个完美的分割超平面），则返回1。
    - 如果达到最大迭代次数后仍未收敛，则认为数据集是线性不可分的，返回-1。

    参数:
    X (np.ndarray): 一个Numpy数组，形状为 (n_samples, n_features + 1)。
                      每一行代表一个数据点。
                      - 前 n_features 列是点的坐标。
                      - 最后一列是类别标签（必须是 1 和 -1）。

    返回:
    int 或 Tuple[int, np.ndarray]:
        - 如果 return_weights=False: 返回 int
            - 1: 如果数据集是线性可分的。
            - -1: 如果数据集不是线性不可分的。
        - 如果 return_weights=True: 返回 tuple (result, weights)
            - result: 1 或 -1 (同上)
            - weights: 权重向量 (如果线性可分) 或 None (如果线性不可分)
    """
    # 检查输入是否合法
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("输入必须是一个至少有两列的Numpy二维数组。")

    # 分离特征和标签
    features = X[:, :-1]
    labels = X[:, -1]
    n_samples, n_features = features.shape

    # 检查是否有至少两个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        if return_weights:
            return (1, None)  # 如果所有点都属于同一个类别，我们视其为线性可分
        return 1

    # 为特征向量添加一个值为1的常数项，以便将偏置b纳入权重向量w
    X_augmented = np.hstack(
        [features, np.ones((n_samples, 1))]
    )  # (n_samples, n_features + 1)

    # 感知机算法实现
    # 初始化增广权重向量 w (包含了偏置b)
    w = np.zeros(n_features + 1)  # (n_features + 1,)

    # 设置最大迭代次数，如果在这个次数内没有收敛，我们就认为它不是线性可分的
    max_epochs = 5000

    for epoch in range(max_epochs):
        misclassified_count = 0

        # 遍历所有数据点
        for i in range(n_samples):
            x_i = X_augmented[i]
            y_i = labels[i]

            # 检查是否误分类
            # 如果 y_i * (w · x_i) <= 0, 则该点被误分类
            if y_i * np.dot(w, x_i) <= 0:
                # 更新权重： w_new = w_old + y_i * x_i
                w = w + y_i * x_i
                misclassified_count += 1

        # 检查是否收敛，如果在一整个epoch中没有任何点被误分类，说明已经找到了分割超平面
        if misclassified_count == 0:
            if return_weights:
                return (1, w)  # 线性可分，返回权重向量
            return 1  # 线性可分

    # 如果在最大迭代次数内算法仍未收敛，则认为不是线性可分的
    if return_weights:
        return (-1, None)
    return -1


def plot_data(X: np.ndarray, title: str, filepath: str, weights: np.ndarray = None) -> None:
    # 创建颜色映射：-1对应红色，1对应蓝色
    colors = ["red" if label == -1 else "blue" for label in X[:, 2]]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7)
    
    # 如果有权重向量，绘制分割线
    if weights is not None and len(weights) == 3:  # 2D情况：w = [w1, w2, b]
        w1, w2, b = weights
        
        # 获取数据点的范围
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # 绘制分割线：w1*x + w2*y + b = 0
        # 即：y = -(w1*x + b) / w2
        if abs(w2) > 1e-10:  # 避免除零
            x_line = np.linspace(x_min, x_max, 100)
            y_line = -(w1 * x_line + b) / w2
            
            # 只绘制在y范围内的部分
            valid_indices = (y_line >= y_min) & (y_line <= y_max)
            if np.any(valid_indices):
                plt.plot(x_line[valid_indices], y_line[valid_indices], 
                        'g-', linewidth=2, label='Separation Line')
                plt.legend()
        else:  # 垂直线情况：w1*x + b = 0
            x_line = -b / w1
            if x_min <= x_line <= x_max:
                plt.axvline(x=x_line, color='green', linewidth=2, label='Separation Line')
                plt.legend()
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return None


def main():
    # 示例 1：线性可分的数据集
    # 分割线: y = x + 1
    # 标签1的点在线上方，标签-1的点在线下方
    X1 = np.array(
        [
            # 标签1的点（在 y = x + 1 上方）
            [-3, 2, 1],   # (-3, 2) 在线上方
            [-1, 3, 1],   # (-1, 3) 在线上方  
            [1, 4, 1],    # (1, 4) 在线上方
            [3, 6, 1],    # (3, 6) 在线上方
            [5, 8, 1],    # (5, 8) 在线上方
            [-4, 0, 1],   # (-4, 0) 在线上方
            [-2, 1, 1],   # (-2, 1) 在线上方
            [0, 3, 1],    # (0, 3) 在线上方
            [2, 5, 1],    # (2, 5) 在线上方
            [4, 7, 1],    # (4, 7) 在线上方
            
            # 标签-1的点（在 y = x + 1 下方）
            [-3, -2, -1], # (-3, -2) 在线下方
            [-1, -1, -1], # (-1, -1) 在线下方
            [1, 0, -1],   # (1, 0) 在线下方
            [3, 1, -1],   # (3, 1) 在线下方
            [5, 2, -1],   # (5, 2) 在线下方
            [-4, -4, -1], # (-4, -4) 在线下方
            [-2, -3, -1], # (-2, -3) 在线下方
            [0, -2, -1],  # (0, -2) 在线下方
            [2, -1, -1],  # (2, -1) 在线下方
            [4, 0, -1],   # (4, 0) 在线下方
        ]
    )
    Y1, weights1 = whetherLinearSeparable_numpy(X1, return_weights=True)
    print(f"对于第一个数据集 X1, 输出为 Y = {Y1} (预期: 1)")
    if weights1 is not None:
        print(f"找到的分割线权重: w1={weights1[0]:.3f}, w2={weights1[1]:.3f}, b={weights1[2]:.3f}")
        print(f"分割线方程: {weights1[0]:.3f}*x + {weights1[1]:.3f}*y + {weights1[2]:.3f} = 0")
    plot_data(X1, "X1(Linear Separable)", "results/X1.png", weights1)

    # 示例 2：线性不可分的数据集
    X2 = np.array(
        [
            [-0.5, -0, -1],
            [3.5, 4.1, -1],
            [4.5, 6, 1],
            [-2, -2.0, -1],
            [-4.1, -2.8, -1],
            [1, 3, -1],
            [-7.1, -4.2, 1],
            [-6.1, -2.2, 1],
            [-4.1, 2.2, 1],
            [1.4, 4.3, 1],
            [-2.4, 4.0, 1],
            [-8.4, -5, 1],
        ]
    )
    Y2, weights2 = whetherLinearSeparable_numpy(X2, return_weights=True)
    print(f"对于第二个数据集 X2, 输出为 Y = {Y2} (预期: -1)")
    plot_data(X2, "X2 (Linear Inseparable)", "results/X2.png", weights2)

    # 示例 3：高维度（3D）线性可分数据集
    X3 = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 1],
            [1, 2, 3, 1],
            [-1, -1, -1, -1],
            [-2, -2, -2, -1],
            [-1, -2, -3, -1],
        ]
    )
    Y3 = whetherLinearSeparable_numpy(X3)
    print(f"对于三维数据集 X3, 输出为 Y = {Y3} (预期: 1)")

    # 示例 4：高维度（3D）线性不可分数据集 (XOR-like)
    X4 = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, -1, 1],
            [-1, 1, -1, 1],
            [-1, -1, 1, 1],
            [-1, -1, -1, -1],
            [-1, 1, 1, -1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
        ]
    )
    Y4 = whetherLinearSeparable_numpy(X4)
    print(f"对于三维数据集 X4, 输出为 Y = {Y4} (预期: -1)")


if __name__ == "__main__":
    main()
