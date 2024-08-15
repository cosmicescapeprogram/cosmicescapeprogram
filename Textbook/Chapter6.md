# 深度學習入門：基於Python的理論與實踐    
## 第6章 與學習相關的技巧
### 6.1參數的更新
最優化：求解最優參數（使損失函數最小）的過程   
方法：+ 隨機梯度下降法（**SGD**）        
#### 6.1.1探險家的故事
**SGD**策略：冒險家通過當前位置的坡度（通過腳底感受地面的傾斜情況），朝著當前位置的坡度最大的方向前進      
#### 6.1.2 SGD        
數學式： 
$$ W \leftarrow W - \eta \frac{\partial L}{\partial W}$$  
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
$$ \nu\leftarrow \alpha\nu - \eta \frac{\partial L}{\partialW}$$  
$$ W\leftarrow W + \nu$$  














     














