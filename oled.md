# oled调试工具           
## 调试方式
+ 串口调试：通过串口通信，将调试信息发送到电脑端，电脑使用串口助手调试信息
+ 显示屏调试：直接将显示屏连接到单片机，将调试信息打印在显示屏上
+ keil调试模式：借助keil软件的调试模式，可使用单步运行、设置断点、查用寄存器及变量等功能
**缩小范围、控制变量、对比测试**   
## 简介
OLED（Organic Light Emitting Diode）：有机发光二极管
优点：功耗低、相应速度快、宽视角、轻薄柔韧            
0.96OLED模块：小巧玲珑、占用接口少、简单易用  
供电：3~5.5V  
分辨率：128*64  
硬件电路：4针脚             
## OLED驱动函数  
+ OLED_Init()  初始化                                  
+ OLED_Clear() 清屏      
+ OLED_ShowChar(1,1,'A') 在第一行的第一列显示字符A          
+ OLED_ShowString（1,2，"HelloWorld） 显示字符串
+ OLED_ShowNum(2,1,12345,5) 显示十进制数字
+ OLED_ShowSignedNum（2,7,-66,2） 显示有符号十进制数字 
+ OLED_ShowHexNum（3,1,0xAA55,4） 显示十六进制数字
+ OLED_ShowBinNum（4,1,0xAA55,16） 显示二进制数字   
# oled显示屏
引脚配置    
```
#define OLED_W_SCL(x)  GPIO_WriteBite(GPIOB,GPIO_Pin_6,(BitAction)(x))
#define OLED_W_SDA(x)  GPIO_WriteBite(GPIOB,GPIO_Pin_9,(BitAction)(x))
```
                            




















