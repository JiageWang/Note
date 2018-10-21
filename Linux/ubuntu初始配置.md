# ubuntu 初始化配置及深度学习环境搭建
---
## 1. 更换软件源
清华源地址：https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
```bash
$ cd /etc/apt/
$ cp sources.list sources.list.backup #制作备份
$ sudo vim sources.list # 修改源列表
```
删除所有内容并更换为(以ubuntu14为例)
```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse
```
## 2. 更改终端配色
```bash
$ gedit ~/.bashrc
```
在末尾添加
```
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35;01m\]\u\[\033[00;00;01m\]@\[\033[01;35;01m\]\h\[\033[00;31;01m\]:\[\033[00;00;01m\]\w \[\033[01;32;01m\]\$ \[\033[01;01;01m\]'
```
## 3. vim安装与配置
* 安装vim
```bash
$ sudo apt-get install vim 
```
* 下载monokai配色
```bash
$ git clone https://github.com/sickill/vim-monokai
$ cp vim-monokai/monokai.vim /usr/share/vim/vim80/colors/  # vim**决定于vim版本
```
* 修改配置文件
```bash
$ vim ~/.vimrc # 修改个人配置文件
```
* 输入配置信息
```
set encoding=utf-8
set fileencoding=utf-8
set fileencodings=ucs-bom,utf-8,chinese,cp936
set guifont=Consolas:h15
language messages zh_CN.utf-8
set lines=45 columns=100
set number  # 显示行号
set autoindent  #自动缩进
set smartindent   #智能缩进
set tabstop=4  #设置tab长度为4个空格
set autochdir

set shiftwidth=4
set foldmethod=manual

syntax enable  #开启语法检测
colorscheme monokai  #主题配色
set nocompatible
set nobackup
```
## 4. git安装与配置
* git安装
```bash
$ sudo apt-get install git
```
* git配置账户
```bash
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```
* 创建git公钥
```bash
$ ssh-keygen -t rsa -C "youremail@example.com"
$ cat ~/.ssh/id_rsa.pub 复制公钥添加到github
```

## 5. pycharm 安装
```bash
$ sudo add-apt-repository ppa:mystic-mirage/pycharm #添加源
$ sudo apt update
$ sudo apt install pycharm-professional #专业版
$ sudo apt install pycharm-community #社区版
```

## 6. sublime安装
```bash
$ sudo add-apt-repository ppa:webupd8team/sublime-text-3 
$ sudo apt-get update 
$ sudo apt-get install sublime-text-installer 
```
## 4. python常用包安装(以python3为例)
* 安装ipython
```bash
$ sudo apt-get install ipython3
```
* 安装pip
```bash
$ sudo apt-get install python3-pip
```
* 基础工具包安装
```bash
$ pip3 install numpy scipy matplotlib pandas sklearn imutils
```
* opencv安装
```bash
$ pip3 install opencv-python
```

## 4. 深度学习环境搭建
* 安装tensorflow
```bash
$ pip3 install tensorflow
```
 * 安装theato
```bash
$ pip3 install theano
```





