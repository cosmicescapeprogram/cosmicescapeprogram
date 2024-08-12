# 深度學習入門：基於Python的理論與實踐   
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
sigmoid函數：$y=\frac{1}{1+exp(-x)}$  
反向傳播為 $\frac{\partial L}{\partial y} y^2exp(-x)$  
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
#### 5.6.2 批版本的Affine層
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





