## 准备工作
添加.gitignore 文件，把不需要追踪的文件和文件夹写进去
## 初始化本地git仓库
`git init`

通过`git status`查看仓库当前状态
通过`git add`添加要追踪的文件
通过`git commit -m 'infofff'`确认添加的文件

首次使用可能需要添加身份
```
git config --global user.name "name"
git config --global user.email "xxx@xxx.com"
```
## 文件添加与修改
![alt text](Image-1.png)  
一个新的文件或新的修改，最先出现在工作区  
通过`git add`把这个修改放进Stage区域  
通过`git commit -m 'xxx'`确认这个修改，放入版本库中

## 撤销修改与版本回退
`git checkout -- file`
1.对于尚未add的即未进暂存区的文件，会返回到与最新版本库一样的版本  
2.对于删除的文件，参见下面 关于删除  
3.对于已经add过但又修改的文件，会返回到最近一次add后的状态  

`git reset HEAD file `  
`git reset --hard HEAD xxx.py #撤销修改与版本回退（简单无脑的方式）`  
如果你连add的修改也要撤回，即不是返回最近一次add状态，    
而是add了，但不想要，想直接返回最新版本库的状态  
这时，你之前的修改会变成unstage状态，即撤销了暂存区，返回到了工作区  

`git reset --hard HEAD file  #把当前文件回退到版本库最新状态`  
`git reset --hard HEAD^ file #把当前文件回退到版本库最新状态的上一个状态`  
同时，git reset命令，也可以回退版本，  
**注意**，命令最后加文件名指只对此文件操作，不加文件名指对工作区所有文件操作
```
git reset --mixed #此为默认方式，不带任何参数的git reset，即使这种方式，它回退到某个版本，只保留源码，回退commit和index信息
git reset --soft #回退到某个版本，只回退了commit的信息，不会恢复到index file一级。如果还要提交，直接commit即可
git reset --hard #彻底回退到某个版本，本地的源码也会变为上一个版本的内容
```
## 删除操作
**删除**  
删除也是一种修改  
在工作区直接rm a.txt和修改了a.txt一样。都需要【类似】git add的操作  
`git rm  #对于删除 类似add的操作就是git rm`  
仅仅rm 对应unstage状态
git rm 对应to be commit状态
git commit 完整删除过程

```
git rm -rf xxx   # 物理删除
git rm -rf --cached xxx  #git删除
```

**撤销删除**  
仅仅rm，直接git checout -- a.txt即可  
git rm 则先git reset HEAD a.txt返回到unstage状态，然后git checout -- a.txt即可  
git rm之后又做了commit，则需要回退到上一版本 git reset HEAD^ a.txt 然后checout  
随时用git status 查看状态，会多有帮助

## 远程仓库
### 添加远程仓库

`git remote add origin https://github.com/xxx/xxx.git #https形式  `  
`git remote add origin git@github.com:xxx/xxx.git #ssh形式  `  
`git push -u origin master #首次推送到远程库`

### 克隆项目
`git clone xxx`

## 分支
### 创建分支
`git checkout -b dev #创建并切换分支`  
`git checkout dev  #切换到dev分支`  
`git checkout -b dev origin/dev  #创建分支并从远程库拉取分支`

### 上传分支
`git push #默认上传master分支或当前分支`  
`git push origin dev #指定上传分支`

### 拉取分支
`git pull #默认拉取master分支或当前分支`  
`git pull origin dev:dev  #指定拉取的分支`

### 删除分支
`git branch -d dev #删除分支`

### 融合分支
#先去到想要做融合的那个分支（不是被融合的分支）
`git checkout master`
`git merge dev  # master 融合了 dev，即dev的内容 merge到了master里面`

## 其他
### diff
`git diff HEAD -- file #查看当前工作区文件和版本库最新文件的不同`

### tag
```
git pull  #先拉取最新
git tag -l “202004*”  #匹配查看202004的标签
git tag -a 20200414 -m “new tag 0414”  #打标签
git push origin 20200414 #推送
git tag -d xx  #删除
git show xxx #详情
```

### 第一次上传GitHub流程
1. 添加远程库  
`git remote add origin https://github.com/xxx/xxx.git #https形式  `  
`git remote add origin git@github.com:xxx/xxx.git #ssh形式  `
2. 本地分支命名  
`git branch -M main`  
`-M # move/rename a branch, even if target exists`
3. 本地push到GitHub  
`git push -u origin main`   
```
-u #相当于执行
git push origin master
git branch --set-upstream master origin/master
```

### ssh形式连接GitHub
如果使用https形式链接GitHub，每次push需要输入GitHub密码，很繁琐  
使用ssh形式会更简单
1. 本地生成ssh密钥
`ssh-keygen -t rsa -C "xxx@xxx.com"`  
生成时会提示密钥位置
2. 把本地公钥上传GitHub
`https://github.com/settings/keys`  
`.pub`文件，复制内容，github上新建key--`New SSH Key`
3. 本地remote改为ssh形式(如有必要)  
`git remote set-url origin git@github.com:username/xxx.git`