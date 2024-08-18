# 深度學習入門：基於Python的理論與實踐  
## 第2章 感知機
### 2.1感知機模型（感知機接收多個輸入信號，輸出一個信號  ）
1）單層感知機    
階躍函數數學式： 

$$
y=
\begin{cases}
0,\omega_1 x_1 +\omega_2 x_2\leqslant0\\
1,\omega_1 x_1 +\omega_2 x_2>0\\ 
\end{cases}
$$ 

**權重起到控製信號流動的作用，偏置則直接調節輸出信號的大小**（*w用於控制各個信號的重要性，b用於控制神經元被激活的容易程度*）  
（感知機可以通過配置適當的權重和偏置，用以模擬不同功能的邏輯電路）

邏輯與： $$\omega_1=\omega_2=0.5$$ ,b=-0.7    
邏輯或： $$\omega_1=\omega_2=0.5$$ ,b=-0.2    
邏輯與非： $$\omega_1=\omega_2=-0.5$$ ,b=0.7    
2）多層感知機（通過疊加，組合多層的感知機，可以劃分非線性的輸出空間，得到更複雜、靈活的表示）  
異或XOR=AND(NAND,OR)    
### 2.2神經網絡模型
#### 2.2.1神經網絡的架構（通過學習，網絡能夠自動調整權重參數和偏置函數）
三層神經網絡：       
+ 輸入層
+ 隱藏層（可多層疊加）
+ 輸出層
#### 2.2.2激活函數（具有非線性能力的特殊函數）：將輸入函數的總和轉換為輸出信號     
（1） 階躍函數 Step function

$$
h=
\begin{cases}
0,\omega_1 x_1 +\omega_2 x_2\leqslant0\\
1,\omega_1 x_1 +\omega_2 x_2>0\\ 
\end{cases}
$$ 

（2）Sigmoid函數

$$h=\frac{1}{1+\exp{(-x)}}$$     

（3）ReLu函數

$$
h=
\begin{cases}
x,x>0\\     
0,x\leqslant0\\ 
\end{cases}
$$ 

（4）Softmax函數  

$$y_k =\frac{\exp{(a_k)}}{\sum_{i=1}^n\exp{(a_i)}}$$

## 第3章 神經網絡                  
## 3.6 手寫數字識別          
### 3.6.1 MNIST數據集        
> MNIST是機器學習領域最有名的數據集之一
>> MNIST數據集的一般使用方法：用訓練圖像進行學習→→學習到的模型度量對測試圖像進行正確分類的程度

load_minst函數的使用：
> 1. load_mnist函数以“(训练图像,训练标签)，(测试图像，测试标签)”的形式返回读入的MNIST数据
> 2. load_mnist(normalize=True,flatten=True, one_hot_label=False) 設置三個參數
>>+ normalize设置是否将输入图像正规化为0.0～1.0的值（如果将该参数设置为 False，则输入图像的像素会保持原来的0～255）
>>+  flatten设置是否展开输入图像（变成一维数组）（如果将该参数设置为 False，则输入图像为1 × 28 × 28的三维数组；若设置为 True，则输入图像会保存为由784个元素构成的一维数组）**显示图像时，需要把它变为原来的28像素 × 28像素的形状。可以通过 reshape()方法的参数指定期望的形状，更改NumPy数组的形状**
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
### 3.6.2神經網絡的推理處理
一、定義函數
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
二、評價識別精度（能在多大程度上正確分類）      
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
+ 正規化：把數據限定到某個範圍內
+ **預處理：對神經網絡的輸入數據進行某種既定的轉換**      
### 3.6.3批處理（打包式的輸入數據）      
> 可以大幅縮短每張圖像的處理時間
批處理代碼 
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
### 3.7小結
神經網絡與感知機的比較：    
+ 相同點：信號的按層傳遞     
+ 不同點：向下一個神經元發送信號時改變信號的激活函數不同
  **神經網絡使用平滑變化的sigmoid函數，感知機使用信號急劇變化的階躍函數**  
## 第4章 神經網絡的*學習*（從訓練數據中自動獲取最優權重參數的過程）  
學習目的：以損失函數為基准，找出使它的值達到最小的權重參數  
### 4.1從數據中學習  
神經網絡特征：從數據中學習（可以由數據自動決定權重參數的值）（現行可分問題可通過有限次數的學習解決，非線性可分問題無法通過自動學習解決）  
#### 4.1.1數據驅動
P83（108）機器學習的方法（神經網絡直接學習圖像本身，圖像中包含的重要特征量也是機器學習的）  
神經網絡優點：對所有問題可以用同樣流程解決（通過不斷學習提供的數據，嘗試發現待求解的問題的模式）  
#### 4.1.2訓練數據和測試數據
數據：訓練數據（監督數據）+測試數據    
1. 使用訓練數據進行學習，尋找最優參數  
2. 使用測試數據評價訓練得到的模型的實際能力    
> 目的：評價模型的**泛化能力**（處理未被觀察過的數據（不包含在訓練數據中的數據）的能力）        

**獲得泛化能力是機器學習的最終目標**  
過擬合:只對某個數據集過度擬合的狀態
### 4.2損失函數（表示神經網絡性能的指標）    
神經網絡的學習通過某個指標表示現在的狀態→→→以這個指標為基准，尋找最優權重參數（該指標為損失函數，可以為任意函數，*一般為均方誤差和交叉熵誤差*）  
#### 4.2.1均方誤差              
均方誤差：
        $E=\frac{1}{2}\sum_{k}(y_k-t_k)^2$；  
        $y_k$表示神經網絡的輸出， $t_k$表示監督數據，k表示數據的維數       
```
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```
### 4.2.2交叉熵誤差       
交叉熵誤差：
          $E=-\sum_k t_k \log y_k$；  
          $y_k$表示神經網絡的輸出， $t_k$是正確解標籤（one-hot表示，只有正確解標籤的索引為1，其他均為0）  
（實際上只計算正確解標籤的輸出的自然對數，交叉熵誤差的值是有正確解標籤所對應的輸出結果決定的）  
```ruby
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```
### 4.2.3 mini-batch學習（利用一部分樣本數據近似地計算總體）  
交叉熵誤差（獲得單個數據的“平均損失函數”）：
$E=-\frac{1}{N} \sum_n \sum_k t_{nk} \log y_{nk}$  
mini-batch學習:從訓練數據中選出一批數據，然後對每個mini-batch進行學習   
從訓練數據中隨機選擇製定個數的數據：np.random.choice()   
### 4.2.4 mini-batch版交叉熵誤差的實現  
```ruby
def cross_entropy_error(y, t):
    if y.ndim == 1: 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```
當監督數據是標簽形式（非one-hot表示）  
```ruby
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
#### 4.2.5為何要設定損失函數    
在進行神經網絡的學習時，不能將識別精度作為指標。因為如果以識別精度為指標，則參數的導數在絕大多數地方都會變為0  
### 4.3數值微分（numerical differentiation梯度法使用梯度的信息決定前進的方向）
#### 4.3.1導數
求函數的導數：   
```ruby
# 不好的实现示例
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h
```
捨入誤差：因省略小數的精細部分的數值（比如，小數點後第8位以後得數值）而造成最終的結果上的誤差  
**中心差分**：以x為中心，計算函數f在（x+h）和（x-h）之間的差分  
**前向差分**：（x+h）和x之間的差分  
**數值微分**：利用微小的差分求導數  
**解析性求解**：基於數學式的推導求導數的過程（得到不含誤差的“真的導數”） 
```ruby
def numerical_diff(f, x):
    h = 1e-4 # 0.0001     
    return (f(x+h) - f(x-h)) / (2*h)
```
#### 4.3.2數值微分的例子
#### 4.3.3偏導數
將多個變量中的某一變量定為目標變量，並將其他變量固定巍峨某個值  
### 4.4梯度（由全部變量的偏導數匯總而成的向量）  
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
**梯度會指向各點處的函數值降低的方向：*梯度指示的方向是各點處的函數值減小最多的方向***         
#### 4.4.1梯度法（巧妙地用梯度來尋找損失函數*最小值*）      
*梯度表示的是各點處的函數值減小最多的方向**無法保證梯度所指的方向就是函數的最小值或真正應該前進的方向***（實際上，在複雜的函數中，梯度指示的方向基本上都不是函數最小處）  
梯度法：通過不斷地沿梯度方向前進，逐漸減小函數值的過程（函數的取值從當前位置沿著梯度方向前進一定距離，然後在新的地方重新求梯度，再沿著新題度方向前進，如此反復，不斷地沿梯度方向前進）  
數學式：  
        $x_0 = x_0 - \eta\frac{\partial f}{\partial x_0}$  
        $x_1 = x_1 - \eta\frac{\partial f}{\partial x_1}$   
$\eta$表示更新量，在神經網絡的學習中，稱為**學習率**（決定一次學習中，應該學習多少，以及在多大程度上更新參數）  
        *需事先確定為某個值*：一般一邊改變學習率的值，一邊確認學習是否正確進行了  
梯度下降法代碼：  
```ruby
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
/*参数 f是要进行最优化的函数,init_x是初始值,lr是学习率learning rate,step_num是梯度法的重复次数,numerical_gradient(f,x)会求函数的梯度，用该梯度乘以学习率得到的值进行更新操作，由 step_num指定重复的次数。*/
```
#### **4.4.2神經網絡的梯度（損失函數關於權重參數的梯度）             
$\frac{\partial L}{\partial W}$ 表示 $\omega$ 稍微變化時，損失函數L會發生多大的變化  
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
### 4.5學習算法的實現
神經網絡的學習步驟（隨機梯度下降法SGD）： 

                  1. mini-batch：從訓練數據中**隨機**選出一部分數據，這部分數據稱為mini-batch，我們的目標是減小mini-batch的損失函數的值  
                  2.計算梯度：為了減小mini-batch的損失函數的值，需要求出各個權重參數的梯度    
                  3.更新參數：將權重參數沿梯度方向進行微小更新    
                  4.重複：重複步驟1、2、3  
#### 4.5.1 2層神經網絡的類
TwolayerNet
#### 4.5.2 mini-batch的實現
#### 4.5.3基於測試數據的評價
## 第5章 誤差傳播法（比數值微分法更搞笑計算權重參數梯度的方法）
### 5.1計算圖
計算圖將計算過程用圖形（數據結構圖）表示出來  
#### 5.1.1用計算圖求解
計算圖通過節點和箭頭表示計算過程  
正向傳播：從左向右進行計算（從計算圖出發點到結束點的傳播）  
反向傳播：從右向左  
#### 5.1.2局部計算  
計算圖特征：可以通過傳遞“局部計算”獲得最終結果  
#### 5.1.3為何用計算圖解題
優點：  
    + 局部計算，簡化問題
    + 能將中間的計算結果保存起來  
    + **可通過反向傳播高效計算導數**  
### 5.2鏈式法則
#### 5.2.1計算圖的反向傳播
#### 5.2.2什麼是鏈式法則
鏈式法則（復合函數的導數的性質）：如果某個函數由復合函數表示，則這個復合函數的導數可以用構成復合函數的各點的導數的乘積表示  
#### 5.2.3鏈式法則和計算圖    
### 5.3反向傳播
#### 5.3.1加法節點的反向傳播
#### 5.3.2乘法節點的反向傳播  
實現乘法節點的反向傳播時，需要保存正向傳播的輸入信號（假髮不需要）
#### 5.3.3蘋果的例子
### 5.4簡單層的實現
#### 5.4.1乘法層的實現MulLayer
#### 5.4.2加法層的實現AddLayer
### 5.5激活層的實現
#### 5.5.1 ReLU層
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
#### 5.5.2 Sigmoid層 
sigmoid函數： $y=\frac{1}{1+exp(-x)}$  
反向傳播為 
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
### 5.6 Affine/Softmax層的實現
#### 5.6.1 Affine層
#### 5.6.2 批版本的Affine（映射）層  
#### 5.6.3 Softmax-with-Loss層
softmax（輸出層函數）會將輸入值正規化後再輸出  
（神經網絡的推理通常不使用Softmax層，但神經網絡的學習階段則需要）  
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
### 5.7誤差反向傳播法的實現 
#### 5.7.1神經網絡學習的全貌圖
神經網絡學習步驟：
                
                1. mini-bitch：從訓練數據中隨機選擇一部分數據  
                2. 計算梯度：**計算損失函數關於各個權重參數的梯度**   
                3. 更新參數：將權重參數沿梯度方向進行微小的更新  
                4. 重複  
#### 5.7.2 對應誤差反向傳播法的神經網絡的實現
**神經網絡的層保存為有序字典OrdereDict**（神經網絡的正向傳播只需按照添加元素的順序調用各層的forward（）方法，反向傳播只需要按照相反的順序調用各層即可）  
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
#### 5.7.3誤差反向傳播法的梯度確認
求梯度的方法：  
            1.數值微分*方法簡單，不易出錯*      
          **2.解析性地求解數學式** （通過誤差反向傳播法在存在大量參數時仍能高效地計算梯度）*計算復雜，容易出錯*  
>**梯度確認**：比較數值微分和誤差反向傳播的結果（確認誤差反向傳播法是否正確）*如果實現正確，誤差是接近0的很小的值*         
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
#### 5.7.4使用誤差反向傳播法的學習 
代碼參考P160
### 5.8小結
## 第6章 與學習相關的技巧
### 6.1參數的更新
最優化：求解最優參數（使損失函數最小）的過程   
方法：隨機梯度下降法（**SGD**）        
#### 6.1.1探險家的故事
**SGD**策略：冒險家通過當前位置的坡度（通過腳底感受地面的傾斜情況），朝著當前位置的坡度最大的方向前進      
#### 6.1.2 SGD        
數學式：  
**W** $$\leftarrow$$ **W** - $$\eta \frac{\partial L}{\partial W}$$  
+ **W**:需要更新的權重參數             
+ $$\frac{\partial L}{\partial W}$$損失函數關於**W**的梯度 
+ $$\eta$$ 表示學習率（會取確定的值）
代碼：
```ruby
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr #lr為learning rate   

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```
optimizer：進行最優化的人（實現參數更新，只需將SGD的參數和梯度的信息傳給optimizer）  
#### 6.1.3 SGD的缺點
SGD的缺點：如果函數的形狀非均向（比如呈延伸狀），搜索的路徑效率會非常低  
SGD低效的原因：梯度的方向沒有指向最小值的方向  
替代：Momentun、AdaGrad、Adam  
#### 6.1.4 Momentum（動量）  
數學式:  
**$$\nu$$** $$\leftarrow \alpha$$ **$$\nu$$** - $$\eta \frac{\partial L}{\partial W}$$  
**W** $$\leftarrow$$ **W** + **$$\nu$$**      
+ **W**:需要更新的權重參數             
+ $$\frac{\partial L}{\partial W}$$損失函數關於**W**的梯度 
+ $$\eta$$ 表示學習率（會取確定的值）
+ **$$\nu$$** 對應物理上的速度
+ $$\alpha$$ **$$\nu$$** 在物體不受任何力時，使物體逐漸減速（ $$\alpha$$設定為0.9之類的值），對應物理上的地面摩擦或空氣阻力
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
行動路徑像小球在碗中滾動，較SGD比較，更快地朝x軸方向靠近，之字程度減輕  
#### 6.1.5 AdaGrad（Ada取自adaptive）  
思想：為參數的每個元素適當地調整學習率（**學習率衰減**：一開始多學，然後逐漸少學）  
數學式：  
**h** $$\leftarrow$$ **h**+ $$\frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}$$  
**W** $$\leftarrow$$ **W** - $$\eta\frac{1}{\sqrt{h}} \frac{\partial L}{\partial W}$$  
+ **W**:需要更新的權重參數               
+ $$\frac{\partial L}{\partial W}$$損失函數關於**W**的梯度 
+ $$\eta$$ 表示學習率（會取確定的值）
+ **h**保存了以前的所有梯度值的平方和（ $$\odot$ $表示對應矩陣元素的乘法） 
參數的元素中變動較大的元素的學習率將變小（按參數的元素進行學習率的衰減，使變動大的參數的學習率逐漸減小）
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
函數的取值高效地向著最小值移動，y軸上的更新程度被減弱，之字形的變動程度衰減  
#### 6.1.6 Adam（將Momentum和AdaGrad融合到一起）  
基於Adam的更新過程像小球在碗中滾動，較Momentum左右晃動的程度有所減輕 
#### 6.1.7使用哪種更新方法呢
不存在能在所有問題中都表現良好的方法，4種方法各有各的特點  
#### 6.1.8基於MNIST數據集的更新方法的比較
與SGD相比，其他3種學習方法更快，速度基本相同，仔細觀察發現AdaGrad學習更快一點  
### 6.2權重的初始值
#### 6.2.1可以將權重初始值設為0嗎
**權重衰減**：以減小權重參數的值為目的進行學習的方法（目的：抑制過擬合、提高泛化能力）  
為了防止“權重均一化”（瓦解權重的對稱結構），必須隨機生成初始值（將權重初始值設為0，將無法進行學習）        
#### 6.2.2隱藏層的激活值的分佈（激活函數最好具有原點對稱性質）  
激活值：激活函數的輸出數據  
梯度消失：偏向0和1的數據分佈會造成反向傳播中梯度的值不斷變小，最後消失  
Xavier初始值：如果前一層的節點數為n，則初始值使用標準差為$$\frac{1}{\sqrt n}$$的分佈（以激活函數是線性函數為前提推到而來）  
#### 6.2.3 ReLu的權重初始值
He初始值（ReLu專用初始值）：當前一層的節點數是n時，使用標準差為$$\sqrt{\frac{2}{n}}$$的高斯分佈  
各層中分佈的廣度相同，因此逆向傳播時也會傳播合適的值     
**當激活函數是ReLu時，權重初始值使用He初始值     
  當激活函數為sigmoid或tanh等S型曲線函數時，初始值使用Xavier初始值**
#### 6.2.4基於MNIST數據集的權重初始值的比較
std=0.01時完全無法學習  
### 6.3 Batch Normalization
Batch Normalization：為了使各層激活函數擁有適當的廣度，“強制性”地調整**激活值**的分佈
#### 6.3.1 Batch Normalization（Batch Norm）的算法  
Batch Norm的優點：
+ 可以使學習快速進行（可以增大學習率）
+ 不那麼依賴初始值（對於初始值不那麼神經質）  
+ 抑制過度擬合（降低Dropout等的必要性）
Batch Norm（向神經網絡中插入對數據分佈進行正規化的層）：以進行學習時的mini-batch為單位，按mini-batch進行正規化

1.正規化數學式：  

$$\mu\leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i$$  

$$\sigma_{B}^{2 }\leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$  

$$\widehat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$  

對輸入數據進行均值為0、方差為1（合適的分佈）的正規化（ $$\varepsilon$$是一個微小值（比如10e-7等），防止出現除以0的情況）  

2.縮放和平移變換數學式：  

$$y_i \leftarrow \gamma \widehat{x_i} + \beta$$  

$$\gamma$$ 和 $$\beta$$ 是參數，一開始 $$\gamma =1$$ ， $$\beta=0$$然後通過學習調整到合適的值
#### 6.3.2 Batch Normalization的評估
發現：使用Batch Norm後，學習進行地更快了；在不使用Batch Norm的情況下，如果不賦予一個尺度好的初始值，學習將無法進行  
結論：Batch Norm可以推動學習的進行，而且對權重初始值變得強壯（不那麼依賴初始值）    
### 6.4正則化（抑制過擬合）  
#### 6.4.1過擬合
發生過擬合的原因：  
+ 模型擁有大量參數、表現力強
+ 訓練數據少
#### 6.4.2權值衰減（抑製過擬合方法）    
原理：在學習過程中對大的權重進行懲罰來抑製過擬合  
方法：為損失函數加上權重的平方范數（L2范數） $$\frac{1}{2} \lamda \W^2$$  →→在求權重梯度的計算中，腰圍之前的誤差反向傳播法的結果加上正則化項的導數$$\lamda W$$  
+ $$\lamda$$ 是控制正則化強度的超參數（設置越大，對大的權重施加的懲罰就越重）  
+ 1/2是能夠使范數的求導結果變為 $$\lamda W$$ 的調整用常量
#### 6.4.3Dropout（網絡模型複雜，只用權值衰減難以應對時）
原理：在學習過程中隨機刪除神經元，停止向前傳遞信號  
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
正向傳播時傳遞了信號的神經元，反向傳播時按原樣傳遞信號；正向傳播時沒有傳遞信號的神經元，反向傳播時信號將停在那裡  
*集成學習：讓多個模型單獨進行學習，推理時再取多個模型的輸出的平均值* 
### 6.5超參數的驗證
超參數：除了權重和偏置等參數，像各層的神經元數量、batch大小、參數更新時的學習率或權重衰減等  
#### 6.5.1驗證數據（不能使用測試數據評估超參數的性能）  
原因：如果使用測試數據調整超參數，超參數的值會對測試數據發生過擬合  
驗證數據（專用確認數據）：用於調整超參數的數據    
**訓練數據用於參數（權重和偏置）的學習，驗證數據用於超參數的性能評估，測試數據用於確認泛化能力（最後使用，且比較理想的是只用一次）**  
分割驗證數據（在此之前打亂輸入數據和監督標籤）  
#### 6.5.2超參數的最優化
縮小超參數的“好值”的存在範圍：一開始先大致設定一個范圍，從這個范圍中隨機選出一個超參數（採樣），用這個採樣到的值進行識別精度的評估，多次重複這個過程  
+ 在神經網絡的超參數的最優化時，隨機采樣的搜索方式比網格搜索等有規律的搜索效果更好（在多個超參數中，各個超參數對最終的識別精度的影響程度不同）
+ 超參數的範圍只要“大致地指定”就可以（以10的階乘的尺度指定范圍（用對數尺度指定））
+ 在超參數的最優化中，深度學習需要很長時間（在超參數的搜索中，需要盡早放棄那些不合邏輯的超參數）
步驟：
+ 步驟0：設定超參數的範圍
+ 步驟1：從設定的超參數范圍中隨機采樣
+ 步驟2：使用步驟1中采樣到的超參數的值進行學習，通過驗證數據評估識別精度（但是要將epoch設置得很小）
+  步驟3：重複步驟1和步驟2（100次等），根據它們的識別精度的結果，縮小超參數的範圍
重複上述步驟，在縮小到一定程度時，從該範圍內選出一個超參數的值
更加精煉的方法：貝葉斯最優化
#### 6.5.3超參數最優化的實現
10**np.random.uniform( ,)  
### 6.6小結
## 第7章卷積神經網絡（CNN，Convolutional Neural Network）
### 7.1整體結構
全連接：相鄰層的所有神經元之間都有連接  
Affine（全連接層）後跟激活函數ReLu層或sigmoid層    
CNN：新增了卷積層（convolution層）和池化層（pooling層）    
連接順序：convolution-relu-（pooling，有時被省略）（Affine-ReLu被替代為convolution-relu）  
+ 靠近輸出的層中使用了之前的Affine-ReLU組合
+ 輸出層使用了之前的Affine-ReLU組合
### 7.2卷積層
#### 7.2.1全連接層存在的問題
**數據的形狀被忽視了** （3維形狀中可能隱藏有值得提取的本質模式，全連接層會忽視形狀，將全部的輸入數據作為相同的神經元（同一維度的神經元）處理，無法利用與形狀相關的信息）  
*卷積層可以保持不變*：當輸入數據是圖像時，卷積層會以3維數據的形式接收輸入數據，並同樣以3維數據的形式輸出至下一層  
+ 特征圖：卷積層的輸入輸出數據    
+ 輸入特征圖：卷積層的輸入數據
+ 輸出特征圖：卷積層的輸出數據
#### 7.2.2卷積計算  
卷積運算（乘積累加運算）：將各個位置上濾波器的元素和輸入的對應元素相乘，然後再求和，將結果保存到輸出的對應位置，將此過程在所有位置都進行一遍  
CNN中，**濾波器的參數對應全連接神經網絡中的權重**，且有偏置（向應用了濾波器的元素加上固定值）         
#### 7.2.3填充
填充（padding）：進行卷積層的處理之前，向輸入數據的周圍填入固定的數據（0）  
目的：調整輸出的大小  
#### 7.2.4步幅（應用濾波器的位置間隔）
增大步幅後，輸出大小會變小；增大填充後，輸出大小會變大  
輸出大小：  

$$OH = \frac{H+2P-FH}{S} +1$$  

$$OH = \frac{W+2P-FW}{S} +1$$

+ 輸入大小為（H,W）  
+ 濾波器大小為（FH,FW）
+ 輸出大小為（OH,OW）
+ 填充為P
+ 步幅為S  
（設定的值必須使公式分別可以除盡）
#### 7.2.5 3維數據的卷積計算
通道方向上有多個特征圖時，會按通道進行輸入數據和濾波器的卷積計算，將結果相加，從而得到輸出     
**在3維數據的卷積計算中，輸入數據和濾波器的通道數要設為相同的值**      
*濾波器大小可以設定為任意值，每個通道的濾波器大小要相同*
#### 7.2.6結合方塊思考  
輸出數據：1張特征圖（通道數為1的特征圖）  
多通道的輸出數據：多個濾波器（權重）FN→處理流             
濾波器的權重數據順序為（output_channel,input_channel,height,width）
#### 7.2.7批處理
將在各層間傳遞的數據保存為4維數據，按（batch_num,channel,height,width）的順序保存數據  
### 7.3池化層  
池化：縮小高、長方向上的空間的運算  
池化的窗口大小會和步幅設定成相同的值  
池化層的特征：  
+ 沒有要學習的參數：只是從目標區域中取最大值或平均值
+ 通道數不發生變化：輸入數據和輸出數據的通道數不會發生變化（計算是按通道獨立進行的）
+ 對微小的位置變化具有魯棒性（強壯）：輸入數據發生微小偏差時，池化仍會返回相同的結果
### 7.4卷積層和池化層的實現
#### 7.4.1 4維數組  
#### 7.4.2基於im2col的展開
NumPy中，訪問元素時最好不要用for語句（處理慢）  
1.im2col（函數）：將輸入數據展開以適合濾波器（對於輸入數據，將應用濾波器的區域橫向展開為1列）  
缺點：消耗更多內存（在濾波器的應用區域重疊的情況下，使用im2col展開後，展開後的元素個數會多於元方塊的元素個數）
2.將卷積層的濾波器縱向展開為1列，並計算2個矩陣的乘積
3.reshape輸出數據
#### 7.4.3卷積層的實現    
卷積層的初始化方法將濾波器（權重）、偏置、步幅、填充作為參數接收
#### 7.4.4池化層的實現
池化的應用區域按通道單獨展開    
步驟：  
1. 展開輸入數據
2. 求各行的最大值
3. 轉換為合適的輸出大小 
### 7.5 CNN的實現
### 7.6 CNN的可視化
### 7.7具有代表性的CNN
