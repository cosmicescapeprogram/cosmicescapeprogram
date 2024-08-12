#  深度學習入門：基於Python的理論與實踐           
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

···
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
···
### 3.6.2神經網絡的推理處理
···   
improt sys,os         
sys.path.append(os.pardir)  
import numpy as np  
from POLimport Image  

def img_show(img):  
    pil_img  
···  




 
