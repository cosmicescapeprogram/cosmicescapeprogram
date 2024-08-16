# 深度學習入門：基於Python的理論與實踐
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
**數據的形狀被忽視了**   
