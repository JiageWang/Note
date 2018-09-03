# 一、Git 本地库

## 1. 配置账户信息
$ git config --global user.name "Your Name"

$ git config --global user.email "email@example.com"

## 2. 新建版本库
$ mkdir learngit

$ cd learngit

$ git init

## 3. 把文件添加到版本库或提交新修改
$ git add <file1>

$ git add <file2>  (把文件修改添加到暂存区)

$ git commit -m <message>  (把暂存区的所有内容提交到当前分支)

## 4. 获取仓库当前的状态
$ git status

## 5. 查看修改内容
$ git diff <file>

## 6. 查看提交历史
$ git log --pretty=oneline

$ git log --graph --pretty=oneline --abbrev-commit (查看分支合并过程)

## 7. 查看命令历史
$ git reflog

## 8. 版本回归
$ git reset --hard <版本号>

## 9. 丢弃工作区的修改
**未add**
$ git checkout -- <file>
**已add 未commit**
$ git reset --hard HEAD

$ git checkout -- <file>

## 10. 从版本库删除文件
$ git rm <file>

$ git commit -m <message>

## 11. 把误删的文件恢复到最新版本
$ git checkout -- <file>

## 12. 暂存工作现场
$ git stash

## 13. 查看保存的工作现场
$ git stash list
---
# 二、Git远程库

## 1. 创建ssh
$ ssh-keygen -t rsa -C "youremail@example.com"
$ cat ~/.ssh/id_rsa.pub 复制公钥添加到github

## 2. 关联远程库
$ git remote add origin git@github.com:username/itemname.git

## 3. 将本地库提交至远程库
$ git push -u origin master (第一次需加-u参数)

## 4. 将远程库拷至本地
$ git clone git@github.com:username/itemname.git

## 5. 获取远程更新
$ git pull origin master

$ git fetch origin
---
# 三、Git分支

## 1. 查看分支：
$ git branch

## 2. 创建分支：
$ git branch <name>

## 3. 切换分支：
$ git checkout <name>

## 4. 创建+切换分支：
$ git checkout -b <name>

## 5. 合并某分支到当前分支：
$ git merge <name>

## 6. 不使用fast-forward方式合并分支:
$ git merge --no-ff -m "merge with no-ff" dev

## 7. 删除分支：
$ git branch -d <name>