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

