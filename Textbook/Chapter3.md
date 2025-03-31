### **第一部分：Linux文件和目录管理命令**
Linux 是基于文件系统的，所有数据都以文件的形式存储在目录结构中，了解基本的文件管理命令是学习 ROS 的基础。

#### **1. `ls`（列出目录内容）**
`ls` 用于列出目录中的文件和文件夹。

- `ls` ：列出当前目录下的所有文件
- `ls -l` ：以详细信息（如权限、所有者、大小等）显示文件列表
- `ls -a` ：显示所有文件，包括隐藏文件（以 `.` 开头的文件）
- `ls -lh` ：文件大小以人类可读的格式（KB、MB）显示

示例：
```bash
ls -lh
```
这会列出当前目录的文件，并且文件大小会以人类可读的方式显示，如 `2.3M` 而不是 `2345678`。

#### **2. `pwd`（显示当前目录）**
`pwd` 显示当前所在的目录路径。

示例：
```bash
pwd
```
如果你当前在 `/home/user` 目录，执行 `pwd` 后会输出：
```
/home/user
```

#### **3. `cd`（切换目录）**
`cd` 用于在不同的目录之间移动。

- `cd 目录名` ：进入指定目录
- `cd ..` ：返回上一级目录
- `cd ~` ：返回用户的主目录（`/home/用户名`）
- `cd -` ：返回上一次所在的目录

示例：
```bash
cd /home/user/Documents  # 进入Documents目录
cd ..                    # 返回上一级目录
```

---

### **第二部分：文件和目录操作命令**
这部分主要介绍如何创建、删除、复制、移动文件和目录。

#### **4. `mkdir`（创建目录）**
`mkdir` 用于创建新目录。

- `mkdir 目录名` ：创建一个新目录
- `mkdir -p 父目录/子目录` ：如果父目录不存在，先创建父目录，再创建子目录

示例：
```bash
mkdir my_folder       # 创建名为 my_folder 的目录
mkdir -p dir1/dir2    # 先创建 dir1 目录，再在其中创建 dir2
```

#### **5. `rmdir`（删除空目录）**
`rmdir` 只能删除**空目录**，如果目录里有文件，则需要用 `rm -r`。

示例：
```bash
rmdir my_folder
```

#### **6. `rm`（删除文件或目录）**
- `rm 文件名` ：删除文件
- `rm -r 目录名` ：递归删除目录及其内容
- `rm -f 文件名` ：强制删除文件，不提示确认
- `rm -rf 目录名` ：强制递归删除目录及其所有内容（危险命令，需谨慎）

示例：
```bash
rm file.txt       # 删除 file.txt 文件
rm -r my_folder   # 递归删除 my_folder 目录及其内容
rm -rf /tmp/data  # 强制删除 /tmp/data 目录（危险！）
```

#### **7. `cp`（复制文件或目录）**
- `cp 源文件 目标路径` ：复制文件到指定路径
- `cp -r 源目录 目标路径` ：复制整个目录及其中的所有文件

示例：
```bash
cp file1.txt /home/user/Documents/  # 复制 file1.txt 到 Documents 目录
cp -r dir1 dir2                     # 复制整个 dir1 目录为 dir2
```

#### **8. `mv`（移动/重命名文件）**
- `mv 源文件 目标路径` ：移动文件
- `mv 旧文件名 新文件名` ：重命名文件
- `mv 旧目录名 新目录名` ：重命名目录

示例：
```bash
mv file1.txt /home/user/Documents/  # 移动 file1.txt 到 Documents 目录
mv old_name.txt new_name.txt        # 重命名文件
mv old_folder new_folder            # 目录改名
```
---

### **第三部分：文件查看命令**
这一部分介绍如何查看文件内容，包括文本文件和日志文件。

#### **9. `cat`（查看文件内容）**
`cat` 用于查看文本文件的内容，适用于小型文件。

- `cat 文件名` ：显示文件内容
- `cat -n 文件名` ：显示时添加行号

示例：
```bash
cat example.txt  # 显示 example.txt 的内容
cat -n example.txt  # 显示 example.txt 的内容，并加上行号
```

#### **10. `tac`（反向查看文件内容）**
`tac` 与 `cat` 相反，它从文件的最后一行开始向上显示。

示例：
```bash
tac example.txt
```

#### **11. `more`（分页查看文件内容）**
当文件内容较长时，可以使用 `more` 进行分页查看。

- `more 文件名` ：逐页查看文件内容
- 按 `空格键` 继续下一页，按 `q` 退出

示例：
```bash
more large_file.txt
```

#### **12. `less`（更灵活的分页查看）**
`less` 和 `more` 类似，但支持**上下滚动**。

- `less 文件名` ：进入文件查看模式
- 按 `↑` / `↓` 滚动
- 按 `q` 退出

示例：
```bash
less example.txt
```

#### **13. `head`（查看文件开头内容）**
`head` 显示文件的前 10 行（默认值），可以指定显示行数。

- `head 文件名` ：显示前 10 行
- `head -n 20 文件名` ：显示前 20 行

示例：
```bash
head example.txt  # 显示 example.txt 的前 10 行
head -n 5 example.txt  # 显示 example.txt 的前 5 行
```

#### **14. `tail`（查看文件结尾内容）**
`tail` 显示文件的最后 10 行（默认值），常用于查看日志文件。

- `tail 文件名` ：显示最后 10 行
- `tail -n 20 文件名` ：显示最后 20 行
- `tail -f 文件名` ：**动态查看文件更新**（常用于监控日志）

示例：
```bash
tail example.txt  # 显示 example.txt 的最后 10 行
tail -f /var/log/syslog  # 实时查看 syslog 日志文件的更新
```
很好，我们继续下一部分！  

---

### **第四部分：文件搜索与定位命令**
这一部分介绍如何在 Linux 系统中查找文件和文本内容。

#### **15. `find`（按条件查找文件）**
`find` 命令用于按**文件名、类型、大小、时间等**条件查找文件。

常见用法：
- `find 目录 -name "文件名"` ：按名称查找文件
- `find 目录 -type d` ：查找目录
- `find 目录 -type f` ：查找文件
- `find 目录 -size +10M` ：查找大于 10MB 的文件
- `find 目录 -mtime -7` ：查找最近 7 天内修改的文件

示例：
```bash
find /home -name "example.txt"  # 在 /home 目录查找名为 example.txt 的文件
find /var/log -size +100M       # 在 /var/log 目录查找大于 100MB 的文件
```

#### **16. `locate`（快速查找文件）**
`locate` 比 `find` 更快，但它依赖于系统的数据库（需要定期更新）。

使用方式：
- `locate 文件名` ：查找文件
- `updatedb` ：手动更新数据库（需要 root 权限）

示例：
```bash
locate example.txt  # 查找 example.txt 文件
```
如果 `locate` 没有找到文件，可以先运行：
```bash
sudo updatedb
```

#### **17. `grep`（搜索文本内容）**
`grep` 用于在**文件内容**中查找特定的字符串。

常见用法：
- `grep "关键词" 文件名` ：在文件中查找关键词
- `grep -r "关键词" 目录` ：递归搜索目录下的所有文件
- `grep -i "关键词" 文件名` ：忽略大小写
- `grep -n "关键词" 文件名` ：显示匹配行的行号

示例：
```bash
grep "error" /var/log/syslog  # 查找 syslog 日志中的 error
grep -r "main()" ~/projects/  # 在 ~/projects 目录中查找包含 main() 的文件
```

#### **18. `which`（查找可执行文件路径）**
`which` 用于查找可执行文件的路径。

示例：
```bash
which python  # 查找 python 命令的位置
which ls      # 查找 ls 命令的位置
```

#### **19. `whereis`（查找命令的所有相关路径）**
`whereis` 不仅能找到命令本身，还能找到它的**源码、文档等**。

示例：
```bash
whereis ls  # 查找 ls 命令的可执行文件、源码和文档
```

#### **20. `type`（查看命令的类型）**
`type` 用于查看某个命令是**内部命令**还是**外部命令**。

示例：
```bash
type ls      # 查看 ls 是内部命令还是外部命令
type cd      # 查看 cd 是内部命令还是外部命令
```

