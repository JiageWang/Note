## 更改bash
* 查看可用bash
```bash
$ cat /etc/shells
```
* 查看当前bash
```bash
$ echo $SHELL
```
* 切换bash(实际更改/etc/passwd)
```bash
$ chsh -s /bin/zsh
```
## 环境变量
* 查看所有环境变量
```bash
$ env 
$ export 
```
* 查看所有环境变量以及自定义变量
```bash
$ set
$ declare
```
其中_变量代表上次输入的命令,自定义变量只能用于当前进程，不能用于子进程等其他进程
* 打印某个变量或字符串
```bash
$ echo $PATH
```
* 将某变量添加至环境变量
```bash
var = "hello"
export var
```
* 将某路径添加至PATH
```bash
export PATH=$PATH:/home/jiage/tools
```

# 别名设置
## 查看所有别名
```bash
$ alias
```
## 设置别名
```bash
$ alias vi="vim"
```
## 删除别名
```bash
$ unalias vi
```
## 使别名重启也有效
修改 `~/.bashrc`,在末尾添加别名设置,例如`alias vi="vim"`,然后
```bash
$ source ~/.bashrc
```




