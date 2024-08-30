# 深度学习入门：基于Python的理论与实践  
## 第2章 感知机
### 2.1感知机模型（感知机接收多个输入信号，输出一个信号  ）
1）单层感知机    
阶跃函数数学式： 

$$
y=
\begin{cases}
0,\omega_1 x_1 +\omega_2 x_2\leqslant0\\
1,\omega_1 x_1 +\omega_2 x_2>0\\ 
\end{cases}
$$ 

**权重起到控制信号流动的作用，偏置则直接调节输出信号的大小**（*w用于控制各个信号的重要性，b用于控制神经元被激活的容易程度*）  
（感知机可以通过配置适当的权重和偏置，用以仿真不同功能的逻辑电路）

逻辑与： $$\omega_1=\omega_2=0.5$$ ,b=-0.7    
逻辑或： $$\omega_1=\omega_2=0.5$$ ,b=-0.2    
逻辑与非： $$\omega_1=\omega_2=-0.5$$ ,b=0.7    
2）多层感知机（通过迭加，组合多层的感知机，可以划分非线性的输出空间，得到更复杂、灵活的表示）  
异或XOR=AND(NAND,OR)    
### 2.2神经网络模型
#### 2.2.1神经网络的架构（通过学习，网络能够自动调整权重参数和偏置函数）
三层神经网络：       
+ 输入层
+ 隐藏层（可多层迭加）
+ 输出层
#### 2.2.2激活函数（具有非线性能力的特殊函数）：将输入函数的总和转换为输出信号     
（1） 阶跃函数 Step function

$$
h=
\begin{cases}
0,\omega_1 x_1 +\omega_2 x_2\leqslant0\\
1,\omega_1 x_1 +\omega_2 x_2>0\\ 
\end{cases}
$$ 

（2）Sigmoid函数

$$h=\frac{1}{1+\exp{(-x)}}$$     

（3）ReLu函数

$$
h=
\begin{cases}
x,x>0\\     
0,x\leqslant0\\ 
\end{cases}
$$ 

（4）Softmax函数  

$$y_k =\frac{\exp{(a_k)}}{\sum_{i=1}^n\exp{(a_i)}}$$

## 第3章 神经网络                  
## 3.6 手写数字识别          
### 3.6.1 MNIST数据集        
> MNIST是机器学习领域最有名的数据集之一
>> MNIST数据集的一般使用方法：用训练图像进行学习→→学习到的模型度量对测试图像进行正确分类的程度

load_minst函数的使用：
> 1. load_mnist函数以“(训练图像,训练标签)，(测试图像，测试标签)”的形式返回读入的MNIST数据
> 2. load_mnist(normalize=True,flatten=True, one_hot_label=False) 设置三个参数
>>+ normalize设置是否将输入图像正规化为0.0～1.0的值（如果将该参数设置为 False，则输入图像的像素会保持原来的0～255）
>>+  flatten设置是否展开输入图像（变成一维数组）（如果将该参数设置为 False，则输入图像为1 × 28 × 28的三维数组；若设置为 True，则输入图像会保存为由784个元素构成的一维数组）**显示图像时，需要把它变为原来的28像素 × 28像素的形状。可以通过 reshape()方法的参数指定期望的形状，更改NumPy数组的形状**
>>+  one_hot_label设置是否将标签保存为one-hot表示（one-hot representation,one-hot表示是仅正确解标签为1，其余皆为0的数组，就像 [0,0,1,0,0,0,0,0,0,0]这样）（当 one_hot_label为 False时，只是像7、2这样简单保存正确解标签；当 one_hot_label为 True时，标签则保存为one-hot表示）
```ruby
import sys, os  
sys.path.append(os.pardir)   
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
def img_show(img):
pil_img = Image.fromarray(np.uint8(img))
pil_img.show()
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False)
img = x_train[0]
label = t_train[0]
print(label) # 5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape) # (28, 28)
img_show(img)
```
### 3.6.2神经网络的推理处理
一、定义函数
```ruby
def get_data():      
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
def init_network():  #读入保存在pickle文件 sample_weight.pkl中的学习到的（以字典变量的形式保存）权重参数  
    with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
    return network
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
```
二、评价识别精度（能在多大程度上正确分类）      
```ruby
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i]) # 以NumPy数组的形式输出各个标签对应的概率   
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```
+ 正规化：把数据限定到某个范围内
+ **预处理：对神经网络的输入数据进行某种既定的转换**      
### 3.6.3批处理（打包式的输入数据）      
> 可以大幅缩短每张图像的处理时间
批处理代码 
```ruby
x, t = get_data()
network = init_network()

**batch_size = 100** # 批数量      
accuracy_cnt = 0

**
for i in range(0, len(x), batch_size):   
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])   
**
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```
### 3.7小结
神经网络与感知机的比较：    
+ 相同点：信号的按层传递     
+ 不同点：向下一个神经元发送信号时改变信号的激活函数不同
  **神经网络使用平滑变化的sigmoid函数，感知机使用信号急剧变化的阶跃函数**  
## 第4章 神经网络的*学习*（从训练数据中自动获取最优权重参数的过程）  
学习目的：以损失函数为基准，找出使它的值达到最小的权重参数  
### 4.1从数据中学习  
神经网络特征：从数据中学习（可以由数据自动决定权重参数的值）（现行可分问题可通过有限次数的学习解决，非线性可分问题无法通过自动学习解决）  
#### 4.1.1数据驱动
P83（108）机器学习的方法（神经网络直接学习图像本身，图像中包含的重要特征量也是机器学习的）  
神经网络优点：对所有问题可以用同样流程解决（通过不断学习提供的数据，尝试发现待求解的问题的模式）  
#### 4.1.2训练数据和测试数据
数据：训练数据（监督数据）+测试数据    
1. 使用训练数据进行学习，寻找最优参数  
2. 使用测试数据评价训练得到的模型的实际能力    
> 目的：评价模型的**泛化能力**（处理未被观察过的数据（不包含在训练数据中的数据）的能力）        

**获得泛化能力是机器学习的最终目标**  
过拟合:只对某个数据集过度拟合的状态
### 4.2损失函数（表示神经网络性能的指针）    
神经网络的学习通过某个指标表示现在的状态→→→以这个指标为基准，寻找最优权重参数（该指针为损失函数，可以为任意函数，*一般为均方误差和交叉熵误差*）  
#### 4.2.1均方误差              
均方误差：
        $E=\frac{1}{2}\sum_{k}(y_k-t_k)^2$；  
        $y_k$表示神经网络的输出， $t_k$表示监督数据，k表示数据的维数       
```
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```
### 4.2.2交叉熵误差       
交叉熵误差：
          $E=-\sum_k t_k \log y_k$；  
          $y_k$表示神经网络的输出， $t_k$是正确解标签（one-hot表示，只有正确解卷标的索引为1，其他均为0）  
（实际上只计算正确解标签的输出的自然对数，交叉熵误差的值是有正确解标签所对应的输出结果决定的）  
```ruby
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```
### 4.2.3 mini-batch学习（利用一部分样本数据近似地计算总体）  
交叉熵误差（获得单个数据的“平均损失函数”）：
$E=-\frac{1}{N} \sum_n \sum_k t_{nk} \log y_{nk}$  
mini-batch学习:从训练数据中选出一批数据，然后对每个mini-batch进行学习   
从训练数据中随机选择制定个数的数据：np.random.choice()   
### 4.2.4 mini-batch版交叉熵误差的实现  
```ruby
def cross_entropy_error(y, t):
    if y.ndim == 1: 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```
当监督数据是标签形式（非one-hot表示）  
```ruby
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
#### 4.2.5为何要设定损失函数    
在进行神经网络的学习时，不能将识别精度作为指标。因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变为0  
### 4.3数值微分（numerical differentiation梯度法使用梯度的信息决定前进的方向）
#### 4.3.1导数
求函数的导数：   
```ruby
# 不好的实现示例
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h
```
舍入误差：因省略小数的精细部分的数值（比如，小数点后第8位以后得数值）而造成最终的结果上的误差  
**中心差分**：以x为中心，计算函数f在（x+h）和（x-h）之间的差分  
**前向差分**：（x+h）和x之间的差分  
**数值微分**：利用微小的差分求导数  
**解析性求解**：基于数学式的推导求导数的过程（得到不含误差的“真的导数”） 
```ruby
def numerical_diff(f, x):
    h = 1e-4 # 0.0001     
    return (f(x+h) - f(x-h)) / (2*h)
```
#### 4.3.2数值微分的例子
#### 4.3.3偏导数
将多个变量中的某一变量定为目标变量，并将其他变量固定巍峨某个值  
### 4.4梯度（由全部变量的偏导数汇总而成的向量）  
```ruby
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同、所有元素都为0的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad
```
**梯度会指向各点处的函数值降低的方向：*梯度指示的方向是各点处的函数值减小最多的方向***         
#### 4.4.1梯度法（巧妙地用梯度来寻找损失函数*最小值*）      
*梯度表示的是各点处的函数值减小最多的方向**无法保证梯度所指的方向就是函数的最小值或真正应该前进的方向***（实际上，在复杂的函数中，梯度指示的方向基本上都不是函数最小处）  
梯度法：通过不断地沿梯度方向前进，逐渐减小函数值的过程（函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度，再沿着新题度方向前进，如此反复，不断地沿梯度方向前进）  
数学式：  
        $x_0 = x_0 - \eta\frac{\partial f}{\partial x_0}$  
        $x_1 = x_1 - \eta\frac{\partial f}{\partial x_1}$   
$\eta$表示更新量，在神经网络的学习中，称为**学习率**（决定一次学习中，应该学习多少，以及在多大程度上更新参数）  
        *需事先确定为某个值*：一般一边改变学习率的值，一边确认学习是否正确进行了  
梯度下降法代码：  
```ruby
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
/*参数 f是要进行最优化的函数,init_x是初始值,lr是学习率learning rate,step_num是梯度法的重复次数,numerical_gradient(f,x)会求函数的梯度，用该梯度乘以学习率得到的值进行更新操作，由 step_num指定重复的次数。*/
```
#### **4.4.2神经网络的梯度（损失函数关于权重参数的梯度）             
$\frac{\partial L}{\partial W}$ 表示 $\omega$ 稍微变化时，损失函数L会发生多大的变化  
```ruby
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
```
### 4.5学习算法的实现
神经网络的学习步骤（随机梯度下降法SGD）： 

                  1. mini-batch：从训练数据中**随机**选出一部分数据，这部分数据称为mini-batch，我们的目标是减小mini-batch的损失函数的值  
                  2.计算梯度：为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度    
                  3.更新参数：将权重参数沿梯度方向进行微小更新    
                  4.重复：重复步骤1、2、3  
#### 4.5.1 2层神经网络的类
TwolayerNet
#### 4.5.2 mini-batch的实现
#### 4.5.3基于测试数据的评价
## 第5章 误差传播法（比数值微分法更搞笑计算权重参数梯度的方法）
### 5.1计算图
计算图将计算过程用图形（数据结构图）表示出来  
#### 5.1.1用计算图求解
计算图通过节点和箭头表示计算过程  
正向传播：从左向右进行计算（从计算图出发点到结束点的传播）  
反向传播：从右向左  
#### 5.1.2局部计算  
计算图特征：可以通过传递“局部计算”获得最终结果  
#### 5.1.3为何用计算图解题
优点：  
    + 局部计算，简化问题
    + 能将中间的计算结果保存起来  
    + **可通过反向传播高效计算导数**  
### 5.2链式法则
#### 5.2.1计算图的反向传播
#### 5.2.2什么是链式法则
链式法则（复合函数的导数的性质）：如果某个函数由复合函数表示，则这个复合函数的导数可以用构成复合函数的各点的导数的乘积表示  
#### 5.2.3链式法则和计算图    
### 5.3反向传播
#### 5.3.1加法节点的反向传播
#### 5.3.2乘法节点的反向传播  
实现乘法节点的反向传播时，需要保存正向传播的输入信号（假发不需要）
#### 5.3.3苹果的例子
### 5.4简单层的实现
#### 5.4.1乘法层的实现MulLayer
#### 5.4.2加法层的实现AddLayer
### 5.5激活层的实现
#### 5.5.1 ReLU层
```ruby
class Relu:
    def __init__(self):
        self.mask = None
/*变量 mask是由 True/False构成的NumPy数组(它会把正向传播时的输入 x的元素中小于等于0的地方保存为 True，其他地方（大于0的元素）保存为 False)*/

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```
#### 5.5.2 Sigmoid层 
sigmoid函数： $y=\frac{1}{1+exp(-x)}$  
反向传播为 
$\frac{\partial L}{\partial y} y^2exp(-x)$  
$\frac{\partial L}{\partial y} y(1-y)$ 
```ruby
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
```
### 5.6 Affine/Softmax层的实现
#### 5.6.1 Affine层
#### 5.6.2 批版本的Affine（映像）层  
#### 5.6.3 Softmax-with-Loss层
softmax（输出层函数）会将输入值正规化后再输出  
（神经网络的推理通常不使用Softmax层，但神经网络的学习阶段则需要）  
```ruby
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 监督数据（one-hot vector）
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
```
### 5.7误差反向传播法的实现 
#### 5.7.1神经网络学习的全貌图
神经网络学习步骤：
                
                1. mini-bitch：从训练数据中随机选择一部分数据  
                2. 计算梯度：**计算损失函数关于各个权重参数的梯度**   
                3. 更新参数：将权重参数沿梯度方向进行微小的更新  
                4. 重复  
#### 5.7.2 对应误差反向传播法的神经网络的实现
**神经网络的层保存为有序字典OrdereDict**（神经网络的正向传播只需按照添加元素的顺序调用各层的forward（）方法，反向传播只需要按照相反的顺序调用各层即可）  
```ruby
def gradient(self, x, t):
    # forward
    self.loss(x, t)

    # backward
    dout = 1
    dout = self.lastLayer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
        dout = layer.backward(dout)

    # 设定    
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return grads
```
#### 5.7.3误差反向传播法的梯度确认
求梯度的方法：  
            1.数值微分*方法简单，不易出错*      
          **2.解析性地求解数学式** （通过误差反向传播法在存在大量参数时仍能高效地计算梯度）*计算复杂，容易出错*  
>**梯度确认**：比较数值微分和误差反向传播的结果（确认误差反向传播法是否正确）*如果实现正确，误差是接近0的很小的值*         
```ruby
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
 
# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

x_batch = x_train[:3]      
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key]-grad_numerical[key]) )
    print(key + ":" + str(diff))
```
#### 5.7.4使用误差反向传播法的学习 
代码参考P160
### 5.8小结
## 第6章 与学习相关的技巧
### 6.1参数的更新
最优化：求解最优参数（使损失函数最小）的过程   
方法：随机梯度下降法（**SGD**）        
#### 6.1.1探险家的故事
**SGD**策略：冒险家通过当前位置的坡度（通过脚底感受地面的倾斜情况），朝着当前位置的坡度最大的方向前进      
#### 6.1.2 SGD        
数学式：  
**W** $$\leftarrow$$ **W** - $$\eta \frac{\partial L}{\partial W}$$  
+ **W**:需要更新的权重参数             
+ $$\frac{\partial L}{\partial W}$$损失函数关于**W**的梯度 
+ $$\eta$$ 表示学习率（会取确定的值）
代码：
```ruby
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr #lr为learning rate   

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```
optimizer：进行最优化的人（实现参数更新，只需将SGD的参数和梯度的信息传给optimizer）  
#### 6.1.3 SGD的缺点
SGD的缺点：如果函数的形状非均向（比如呈延伸状），搜索的路径效率会非常低  
SGD低效的原因：梯度的方向没有指向最小值的方向  
替代：Momentun、AdaGrad、Adam  
#### 6.1.4 Momentum（动量）  
数学式:  
**$$\nu$$** $$\leftarrow \alpha$$ **$$\nu$$** - $$\eta \frac{\partial L}{\partial W}$$  
**W** $$\leftarrow$$ **W** + **$$\nu$$**      
+ **W**:需要更新的权重参数             
+ $$\frac{\partial L}{\partial W}$$损失函数关于**W**的梯度 
+ $$\eta$$ 表示学习率（会取确定的值）
+ **$$\nu$$** 对应物理上的速度
+ $$\alpha$$ **$$\nu$$** 在物体不受任何力时，使物体逐渐减速（ $$\alpha$$设定为0.9之类的值），对应物理上的地面摩擦或空气阻力
```ruby
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

    for key in params.keys():
        self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
        params[key] += self.v[key]
```
行动路径像小球在碗中滚动，较SGD比较，更快地朝x轴方向靠近，之字程度减轻  
#### 6.1.5 AdaGrad（Ada取自adaptive）  
思想：为参数的每个元素适当地调整学习率（**学习率衰减**：一开始多学，然后逐渐少学）  
数学式：  
**h** $$\leftarrow$$ **h**+ $$\frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}$$  
**W** $$\leftarrow$$ **W** - $$\eta\frac{1}{\sqrt{h}} \frac{\partial L}{\partial W}$$  
+ **W**:需要更新的权重参数               
+ $$\frac{\partial L}{\partial W}$$损失函数关于**W**的梯度 
+ $$\eta$$ 表示学习率（会取确定的值）
+ **h**保存了以前的所有梯度值的平方和（ $$\odot$ $表示对应矩阵元素的乘法） 
参数的元素中变动较大的元素的学习率将变小（按参数的元素进行学习率的衰减，使变动大的参数的学习率逐渐减小）
```ruby
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```
函数的取值高效地向着最小值移动，y轴上的更新程度被减弱，之字形的变动程度衰减  
#### 6.1.6 Adam（将Momentum和AdaGrad融合到一起）  
基于Adam的更新过程像小球在碗中滚动，较Momentum左右晃动的程度有所减轻 
#### 6.1.7使用哪种更新方法呢
不存在能在所有问题中都表现良好的方法，4种方法各有各的特点  
#### 6.1.8基于MNIST数据集的更新方法的比较
与SGD相比，其他3种学习方法更快，速度基本相同，仔细观察发现AdaGrad学习更快一点  
### 6.2权重的初始值
#### 6.2.1可以将权重初始值设为0吗
**权重衰减**：以减小权重参数的值为目的进行学习的方法（目的：抑制过拟合、提高泛化能力）  
为了防止“权重均一化”（瓦解权重的对称结构），必须随机生成初始值（将权重初始值设为0，将无法进行学习）        
#### 6.2.2隐藏层的激活值的分布（激活函数最好具有原点对称性质）  
激活值：激活函数的输出数据  
梯度消失：偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失  
Xavier初始值：如果前一层的节点数为n，则初始值使用标准偏差为$$\frac{1}{\sqrt n}$$的分布（以激活函数是线性函数为前提推到而来）  
#### 6.2.3 ReLu的权重初始值
He初始值（ReLu专用初始值）：当前一层的节点数是n时，使用标准偏差为$$\sqrt{\frac{2}{n}}$$的高斯分布  
各层中分布的广度相同，因此逆向传播时也会传播合适的值     
**当激活函数是ReLu时，权重初始值使用He初始值     
  当激活函数为sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值**
#### 6.2.4基于MNIST数据集的权重初始值的比较
std=0.01时完全无法学习  
### 6.3 Batch Normalization
Batch Normalization：为了使各层激活函数拥有适当的广度，“强制性”地调整**激活值**的分布
#### 6.3.1 Batch Normalization（Batch Norm）的算法  
Batch Norm的优点：
+ 可以使学习快速进行（可以增大学习率）
+ 不那么依赖初始值（对于初始值不那么神经质）  
+ 抑制过度拟合（降低Dropout等的必要性）
Batch Norm（向神经网络中插入对数据分布进行正规化的层）：以进行学习时的mini-batch为单位，按mini-batch进行正规化

1.正规化数学式：  

$$\mu\leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i$$  

$$\sigma_{B}^{2 }\leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$  

$$\widehat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$  

对输入数据进行均值为0、方差为1（合适的分布）的正规化（ $$\varepsilon$$是一个微小值（比如10e-7等），防止出现除以0的情况）  

2.缩放和平移变换数学式：  

$$y_i \leftarrow \gamma \widehat{x_i} + \beta$$  

$$\gamma$$ 和 $$\beta$$ 是参数，一开始 $$\gamma =1$$ ， $$\beta=0$$然后通过学习调整到合适的值
#### 6.3.2 Batch Normalization的评估
发现：使用Batch Norm后，学习进行地更快了；在不使用Batch Norm的情况下，如果不赋予一个尺度好的初始值，学习将无法进行  
结论：Batch Norm可以推动学习的进行，而且对权重初始值变得强壮（不那么依赖初始值）    
### 6.4正则化（抑制过拟合）  
#### 6.4.1过拟合
发生过拟合的原因：  
+ 模型拥有大量参数、表现力强
+ 训练数据少
#### 6.4.2权值衰减（抑制过拟合方法）    
原理：在学习过程中对大的权重进行惩罚来抑制过拟合  
方法：为损失函数加上权重的平方范数（L2范数） $$\frac{1}{2} \lambda W^2$$  →→在求权重梯度的计算中，要为之前的误差反向传播法的结果加上正则化项的导数 $$\lambda W$$  
+ $$\lambda$$ 是控制正则化强度的超参数（设置越大，对大的权重施加的惩罚就越重）  
+ 1/2是能够使范数的求导结果变为 $$\lambda W$$ 的调整用常量
#### 6.4.3Dropout（网络模型复杂，只用权值衰减难以应对时）
原理：在学习过程中随机删除神经元，停止向前传递信号  
```ruby
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  #每次正向传播时，self.mask中都会以 False的形式保存要删除的神经元
             return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask
```
正向传播时传递了信号的神经元，反向传播时按原样传递信号；正向传播时没有传递信号的神经元，反向传播时信号将停在那里  
*集成学习：让多个模型单独进行学习，推理时再取多个模型的输出的平均值* 
### 6.5超参数的验证
超参数：除了权重和偏置等参数，像各层的神经元数量、batch大小、参数更新时的学习率或权重衰减等  
#### 6.5.1验证数据（不能使用测试数据评估超参数的性能）  
原因：如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合  
验证数据（专用确认数据）：用于调整超参数的数据    
**训练数据用于参数（权重和偏置）的学习，验证数据用于超参数的性能评估，测试数据用于确认泛化能力（最后使用，且比较理想的是只用一次）**  
分割验证数据（在此之前打乱输入数据和监督卷标）  
#### 6.5.2超参数的最优化
缩小超参数的“好值”的存在范围：一开始先大致设定一个范围，从这个范围中随机选出一个超参数（采样），用这个采样到的值进行识别精度的评估，多次重复这个过程  
+ 在神经网络的超参数的最优化时，随机采样的搜索方式比网格搜索等有规律的搜索效果更好（在多个超参数中，各个超参数对最终的识别精度的影响程度不同）
+ 超参数的范围只要“大致地指定”就可以（以10的阶乘的尺度指定范围（用对数尺度指定））
+ 在超参数的最优化中，深度学习需要很长时间（在超参数的搜索中，需要尽早放弃那些不合逻辑的超参数）
步骤：
+ 步骤0：设定超参数的范围
+ 步骤1：从设定的超参数范围中随机采样
+ 步骤2：使用步骤1中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将epoch设置得很小）
+  步骤3：重复步骤1和步骤2（100次等），根据它们的识别精度的结果，缩小超参数的范围
重复上述步骤，在缩小到一定程度时，从该范围内选出一个超参数的值
更加精炼的方法：贝叶斯最优化
#### 6.5.3超参数最优化的实现
10**np.random.uniform( ,)  
### 6.6小结
## 第7章卷积神经网络（CNN，Convolutional Neural Network）
### 7.1整体结构
全连接：相邻层的所有神经元之间都有连接  
Affine（全连接层）后跟激活函数ReLu层或sigmoid层    
CNN：新增了卷积层（convolution层）和池化层（pooling层）    
连接顺序：convolution-relu-（pooling，有时被省略）（Affine-ReLu被替代为convolution-relu）  
+ 靠近输出的层中使用了之前的Affine-ReLU组合
+ 输出层使用了之前的Affine-ReLU组合
### 7.2卷积层
#### 7.2.1全连接层存在的问题
**数据的形状被忽视了** （3维形状中可能隐藏有值得提取的本质模式，全连接层会忽视形状，将全部的输入数据作为相同的神经元（同一维度的神经元）处理，无法利用与形状相关的信息）  
*卷积层可以保持不变*：当输入数据是图像时，卷积层会以3维数据的形式接收输入数据，并同样以3维数据的形式输出至下一层  
+ 特征图：卷积层的输入输出数据    
+ 输入特征图：卷积层的输入数据
+ 输出特征图：卷积层的输出数据
#### 7.2.2卷积计算  
卷积运算（乘积累加运算）：将各个位置上滤波器的元素和输入的对应元素相乘，然后再求和，将结果保存到输出的对应位置，将此过程在所有位置都进行一遍  
CNN中，**滤波器的参数对应全连接神经网络中的权重**，且有偏置（向应用了滤波器的元素加上固定值）         
#### 7.2.3填充
填充（padding）：进行卷积层的处理之前，向输入数据的周围填入固定的数据（0）  
目的：调整输出的大小  
#### 7.2.4步幅（应用滤波器的位置间隔）
增大步幅后，输出大小会变小；增大填充后，输出大小会变大  
输出大小：  

$$OH = \frac{H+2P-FH}{S} +1$$  

$$OH = \frac{W+2P-FW}{S} +1$$

+ 输入大小为（H,W）  
+ 滤波器大小为（FH,FW）
+ 输出大小为（OH,OW）
+ 填充为P
+ 步幅为S  
（设定的值必须使公式分别可以除尽）
#### 7.2.5 3维数据的卷积计算
通道方向上有多个特征图时，会按信道进行输入数据和滤波器的卷积计算，将结果相加，从而得到输出     
**在3维数据的卷积计算中，输入数据和滤波器的信道数要设为相同的值**      
*滤波器大小可以设定为任意值，每个通道的滤波器大小要相同*
#### 7.2.6结合方块思考  
输出数据：1张特征图（通道数为1的特征图）  
多信道的输出数据：多个滤波器（权重）FN→处理流             
滤波器的权重数据顺序为（output_channel,input_channel,height,width）
#### 7.2.7批处理
将在各层间传递的数据保存为4维数据，按（batch_num,channel,height,width）的顺序保存数据  
### 7.3池化层  
池化：缩小高、长方向上的空间的运算  
池化的窗口大小会和步幅设定成相同的值  
池化层的特征：  
+ 没有要学习的参数：只是从目标区域中取最大值或平均值
+ 通道数不发生变化：输入数据和输出数据的信道数不会发生变化（计算是按通道独立进行的）
+ 对微小的位置变化具有鲁棒性（强壮）：输入数据发生微小偏差时，池化仍会返回相同的结果
### 7.4卷积层和池化层的实现
#### 7.4.1 4维数组  
#### 7.4.2基于im2col的展开
NumPy中，访问元素时最好不要用for语句（处理慢）  
1.im2col（函数）：将输入数据展开以适合滤波器（对于输入数据，将应用滤波器的区域横向展开为1列）  
缺点：消耗更多内存（在滤波器的应用区域重迭的情况下，使用im2col展开后，展开后的元素个数会多于元方块的元素个数）
2.将卷积层的滤波器纵向展开为1列，并计算2个矩阵的乘积
3.reshape输出数据
#### 7.4.3卷积层的实现    
卷积层的初始化方法将滤波器（权重）、偏置、步幅、填充作为参数接收
#### 7.4.4池化层的实现
池化的应用区域按信道单独展开    
步骤：  
1. 展开输入数据
2. 求各行的最大值
3. 转换为合适的输出大小 
### 7.5 CNN的实现
### 7.6 CNN的可视化
### 7.7具有代表性的CNN
