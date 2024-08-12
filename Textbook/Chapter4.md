# 深度學習入門：基於Python的理論與實踐  
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
        $y_k$表示神經網絡的輸出，$t_k$表示監督數據，k表示數據的維數       
```
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```
### 4.2.2交叉熵誤差       
交叉熵誤差：
          $E=-\sum_k t_k \log y_k$；  
          $y_k$表示神經網絡的輸出，$t_k$是正確解標籤（one-hot表示，只有正確解標籤的索引為1，其他均為0）  
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











          











