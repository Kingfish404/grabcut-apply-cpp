# GrabCut Algorithm

GrabCut是一种交互式迭代前景提取算法，这个项目是对其的实践，参考opencv的源码

核心的点
- 混合高斯模型
- 迭代图像分割
- Graph Cut
- border matting: 这个不是重点

opencv mask四状态  
- 背景`GCD_BGD`：0  
- 前景`GCD_FGD`：1
- 可能的背景`GCD_PR_BGD`：2
- 可能的前景`GCD_PR_FGD`：3

## 代码

* main.cpp: 主要的用户交互逻辑  
* grabcut.hpp: grabcut算法的实现（含注释）及其依赖的GMM等  
* grabcut_ref.cpp: opencv中的样例的修改  

## 目录

- images: 测试用的图像文件
- lib: 可能用到的库
- pages: 算法的相关paper

## DEMO

框选并提取出 可能的前景`GCD_PR_FGD`，未被框则选为 背景`GCD_BGD`，按`n`获得初次分割结果

![](./images/demo-1.png)

左键点击选取`GCD_FGD`，右键点击选取`GCD_BGD`，按`n`继续迭代更新

![](./images/demo-2.png)

