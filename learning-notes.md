# STM32 Learning Notes   
### STM32F103              
Core：ARM Cortex-M3  
Frequency: 72MHZ 
RAM: 20K(SRAM)  
ROM: 64K(FLASH)  
Voltage: 2.0V~3.6V **(Standard 3.3V)**  
封装：LQFP48  
#### Periphearl(外设）
NVIC 嵌套向量中断控制器  
Systick 系统滴答定时器  
RCC 复位和时钟控制  
GPIO TIM 定时器  
EXIT 外部中断  
DMA 直接内存访问  
I2C I2C通信  
SPI SPI通信  
#### STM32F1系列系统架构
课本P69 DMA做一些简单且须重复的工作，例如：数据搬运  
#### 启动配置
BOOT1 BOOT1   (BOOT引脚的值只在上电之后的一刻有效，四个时钟之后失效）         
X     0        主闪存存储器     主闪存存储器被选为启动区域
0     1        系统存储器        系统存储器被选为启动区域（做串口下载使用，没有STLINK或LINK）    
1     1        内置SRAM         内置SRAM被选为启动区域

**如果要让引脚正常工作，应先设置好电源部分和最小系统部分的引脚** 
#### 最小系统电路
在单片机中只有一个芯片一般无法工作，需要连接最基本的电路，这个最基本的电路就叫做最小系统电路
如果需要接备用电池，可以接一个3V的纽扣电池，备用电池一般为RTC和备份寄存器服务



