### C语言数据类型        
注意事项：在51中int占16位，而在STM32中int占32位。如果要用16位数据，要用short来表示（同long、unsinged long）  
stdint关键字     
int8_t char，表示8位整型数据（uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t）  
### C语言宏定义       
#define 用一个字符串代替一个数字，便于理解，防止出错；提取程序中经常出现的参数，便于快速修改  
### C语言typedef      
typedef 将一个比较长的**变量类型**名换个名字，便于使用（typedef unsigned char uint8_t**;**) 
### C语言结构体
用途：数据打包，不同类型变量的集合  
struct{char x;int y;float z;}c;   
typedef struct{char x;int y;float z;} StructName_t;  
### C语言枚举
enum 定义一个取值受限制的整型变量，用于限制变量取值范围；宏定义的集合  

