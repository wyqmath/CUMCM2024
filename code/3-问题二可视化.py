import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义粒子类
class Particle:
    def __init__(self, n_variables):
        # 初始化粒子位置为[-5, 5]之间的随机数
        self.position = np.random.rand(n_variables) * 12 - 25  # 初始化粒子位置 (-5到5的随机数)
        # 初始化粒子速度为0到0.01之间的随机数
        self.velocity = np.random.rand(n_variables) * 0.01  # 初始化粒子速度
        # 记录个体最优位置
        self.best_position = self.position.copy()  # 记录个体最优位置
        # 记录个体最优适应度值，初始为负无穷
        self.best_value = float('-inf')  # 记录个体最优适应度值
        # 当前适应度值，初始为负无穷
        self.current_value = float('-inf')  # 当前适应度值

# 定义目标函数
def objective_function(x):
    """模拟目标函数，用于评估粒子适应度"""
    # 计算目标函数值，返回负平方和加上噪声
    return -np.sum(x ** 4 + x ** 2 + np.random.randn() * 0.1)  # 负平方和 + 噪声

# 定义粒子群优化类
class PSO:
    def __init__(self, n_particles, n_variables, max_iter, w=0.5, c1=2, c2=2):
        # 初始化粒子数量、变量数量和最大迭代次数
        self.n_particles = n_particles  # 粒子数量
        self.n_variables = n_variables  # 问题维度
        self.max_iter = max_iter  # 最大迭代次数
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 群体学习因子
        # 初始化粒子群
        self.particles = [Particle(n_variables) for _ in range(n_particles)]  # 初始化粒子群
        # 初始化全局最优位置
        self.gbest_position = np.random.rand(n_variables) * 10 - 5  # 初始化全局最优位置
        # 初始化全局最优值
        self.gbest_value = float('-inf')  # 初始化全局最优值
        # 用于记录每次迭代的全局最优值
        self.history = []  # 用于记录每次迭代的全局最优值

    # 优化过程
    def optimize(self):
        for iteration in range(self.max_iter):  # 遍历每次迭代
            for particle in self.particles:  # 遍历每个粒子
                # 计算当前适应度值
                particle.current_value = objective_function(particle.position)  # 计算当前适应度值

                # 更新个体最优位置
                if particle.current_value > particle.best_value:
                    particle.best_value = particle.current_value  # 更新个体最优适应度值
                    particle.best_position = particle.position.copy()  # 更新个体最优位置

                # 更新全局最优位置
                if particle.current_value > self.gbest_value:
                    self.gbest_value = particle.current_value  # 更新全局最优适应度值
                    self.gbest_position = particle.position.copy()  # 更新全局最优位置

            # 更新粒子速度和位置
            for particle in self.particles:
                inertia = self.w * particle.velocity  # 计算惯性部分
                cognitive = self.c1 * np.random.rand(self.n_variables) * (particle.best_position - particle.position)  # 计算个体学习部分
                social = self.c2 * np.random.rand(self.n_variables) * (self.gbest_position - particle.position)  # 计算群体学习部分
                particle.velocity = inertia + cognitive + social  # 更新速度
                particle.position += particle.velocity  # 更新位置

            # 记录当前全局最优值
            self.history.append(self.gbest_value)

            # 可视化粒子群
            self.visualize(iteration)

    # 可视化每次迭代的粒子分布
    def visualize(self, iteration):
        plt.clf()  # 清空当前图像
        particle_positions = np.array([p.position for p in self.particles])  # 获取所有粒子的位置
        plt.scatter(particle_positions[:, 0], particle_positions[:, 1], color='blue', label='Particles', s=50)  # 绘制粒子
        plt.scatter(self.gbest_position[0], self.gbest_position[1], color='red', label='Global Best', s=200, edgecolors='black')  # 绘制全局最优位置
        plt.title(f'Iteration {iteration + 1}')  # 设置标题
        plt.xlabel('Variable 1')  # 设置x轴标签
        plt.ylabel('Variable 2')  # 设置y轴标签
        plt.xlim(-6, 6)  # 设置x轴范围
        plt.ylim(-6, 6)  # 设置y轴范围
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        plt.pause(0.1)  # 暂停以更新图像

# 运行粒子群优化
np.random.seed(500)  # 固定随机种子保证结果可复现

n_particles = 50  # 粒子数量
n_variables = 2  # 问题维度（简化为2维，用于可视化）
max_iter = 100  # 最大迭代次数

pso = PSO(n_particles, n_variables, max_iter)  # 创建PSO实例
plt.figure(figsize=(8, 6))  # 设置图形大小
pso.optimize()  # 开始优化

# 绘制优化过程中的全局最优值变化
plt.figure(figsize=(8, 6))  # 设置图形大小
plt.plot(pso.history, color='green', linewidth=2)  # 绘制全局最优值变化曲线
plt.title('Global Best Value Over Iterations', fontsize=14)  # 设置标题
plt.xlabel('Iteration', fontsize=12)  # 设置x轴标签
plt.ylabel('Global Best Value', fontsize=12)  # 设置y轴标签
plt.grid(True)  # 显示网格
plt.show()  # 显示图形
