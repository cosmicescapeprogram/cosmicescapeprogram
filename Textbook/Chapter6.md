# 深度學習入門：基於Python的理論與實踐    
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

