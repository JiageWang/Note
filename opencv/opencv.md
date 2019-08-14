## 一、图像基本操作

### 1. 图片读取

读取成功返回像素矩阵，读取失败返回None且**不报错**

- **cv2.IMREAD_COLOR** ： 以彩色模式读取
- **cv2.IMREAD_GRAYSCALE**：# 以灰度模式读取

```python
img = cv2.imread('file_name') # 默认读取
img_color = cv2.imread('file_name', cv2.IMREAD_COLOR)  
img_gray = cv2.imread('file_name', cv2.IMREAD_GRAYSCALE)    
```

### 2. 图片显示

```python
cv2.imshow('window_name', img)  # 显示图片
cv2.waitKey(0) # 设置等待时间
cv2.destroyAllWindows()  # 销毁所有窗口
```

### 3. 图片保存

```python
cv2.imwrite('file_name', img)
```

### 4. 视频流读取与显示

```python
cap = cv2.VideoCapture('filename')  # 读取视频文件
cap = cv2.VideoCapture(0)  # 读取0号摄像头

if cap.isOpened():  # 判断是否成功打开v
    while True:
        ret, frame = cap.read()  # 读取下一帧，ret为读取状态，成功读取为True，frame为帧图像，读取失败时
        if not ret:
            break  # 读取失败时退出
        cv2.imshow('test.mp4', frame)  # 显示当前帧
        if cv2.waitKey(1) == ord('q'):  # 若按下q退出
            break
```

### 5. 图像属性

```python
img.shape # 返回图像大小与通道数
img.size  # 像素矩阵总大小
img.dtype  # 像素矩阵的数据类型
```

### 6. 图像截取（Roi区域）

```python
img[:200, :500, :] # 与矩阵截取一致
```

### 7. 通道分离与合并

- 通道分离

```python
b, g, r = cv2.split(img)  # 使用函数分离
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]  # 使用切片分离
```

- 通道合并

```python
img = cv2.merge((b,g,r))
```

- 显示单通道

```python
cur_img = img.copy()
cur_img[:, :, 1] = 0  # g
cur_img[:, :, 2] = 0  # r
cv2.show("b", cur_img)
```

### 8. 边界填充

五种填充模式

- **cv2.BORDER_REPLICATE**：复制法，也就是复制最边缘像素。
- **cv2.BORDER_REFLECT**：反射法，对图像中的像素在两边进行复制，例如：cba|abc|cba
- **cv2.BORDER_REFLECT_101**：反射法，也就是以最边缘像素为轴对称，例如：dcb|abcd|cba
- **cv2.BORDER_WRAP**：外包装法， cdefgh|abcdefgh|abcdefg
- **cv2.BORDER_CONSTANT**：常量法，常数值填充。

```python
t, b, l, r = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, t, b, l, r, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
```

![padding](C:\Users\Administrator\OneDrive\笔记\opencv\padding.png)

### 9. 图像放缩

```python
img = cv2.resize(img, (400, 400)) # 将图片放缩至400x400
img = cv2.resize(img, (0, 0), fx=1.5, fy=2) # 将图片放缩至1.5倍大小
```

### 10. 数值计算

- 直接使用矩阵相加，对于相加后超过255的值，进行 value % 256 操作

```python
img1 = cv2.imread('cat.jpg')
img2 = cv2.imread('cat.jpg')
img3 = img1 + img2  # shape需要相同
print(img1[50:52, 50:52, 0])  # [[133 120],[166 130]]
print(img2[50:52, 50:52, 0])  # [[133 120],[166 130]]
print(img3[50:52, 50:52, 0])  # [[ 10 240],[ 76   4]]
```

- 使用cv2.add()函数，对于相加后超过255的值，直接当作255

```python
img1 = cv2.imread('cat.jpg')
img2 = cv2.imread('cat.jpg')
img3 = cv2.add(img1, img2)  # shape需要相同
print(img1[50:52, 50:52, 0])  # [[133 120],[166 130]]
print(img2[50:52, 50:52, 0])  # [[133 120],[166 130]]
print(img3[50:52, 50:52, 0])  # [[255 240],[255 255]]
```

- 按位与

```python
mask = np.zeros_like(img[::0])
mask[100:300, 100:300] = 255
cv2.bitwise_and(img, img, mask=mask)
```



### 11. 图像融合

img1与img2的形状需要相同，$ img = img1 \times 0.4 + img2 \times 0.6 + 0 $

```python
img = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
```

### 12. 图像拼接

- 横向拼接

```python
img = np.hstack((img, img, img))
```

- 纵向拼接

```python
img = np.hstack((img, img, img))
```



## 二、阈值操作

### 1. 阈值处理 cv2.threshold(）

```python
ret, dst = cv2.threshold(src, thresh, maxval, method)
```

#### 函数参数

* `src`：原图像

* `thresh`：阈值

* `maxval`：阈值分割后得到图像的最大值

* `method`：操作方法，主要有以下六种，可使用`|`操作符叠加使用

  * `cv2.THRESH_BINARY`：二进制阈值化，大于`thresh`元素处理为`maxval`，其余为0

    ![Threshold_Tutorial_Theory_Binary](C:\Users\Administrator\OneDrive\笔记\opencv\Threshold_Tutorial_Theory_Binary.png)

  * `cv2.THRESH_BINARY_INV`：反二进制阈值化，小于`thresh`元素处理为`maxval`，其余为0

    ![Threshold_Tutorial_Theory_Binary_Inverted](C:\Users\Administrator\OneDrive\笔记\opencv\Threshold_Tutorial_Theory_Binary_Inverted.png)

  * `cv2.THRESH_TOZERO`：阈值化为0，小于`thresh`元素处理为0，其余不变

    ![Threshold_Tutorial_Theory_Zero](C:\Users\Administrator\OneDrive\笔记\opencv\Threshold_Tutorial_Theory_Zero.png)

  * `cv2.THRESH_TOZERO_INV`：反阈值化为0，大于`thresh`元素处理为0，其余不变

    ![Threshold_Tutorial_Theory_Zero_Inverted](C:\Users\Administrator\OneDrive\笔记\opencv\Threshold_Tutorial_Theory_Zero_Inverted.png)

  * `cv2.THRESH_TRUNC`：截断阈值化，大于`thresh`部分处理为`thresh`，其余不变

    ![Threshold_Tutorial_Theory_Truncate](C:\Users\Administrator\OneDrive\笔记\opencv\Threshold_Tutorial_Theory_Truncate.png)

  * `cv2.THRESH_OTSU`：大津算法，自适应阈值，将`thresh`设置为0

#### 返回值

* `ret`：返回阈值

* `dst`：处理结果

#### 实例效果

##### 二进制阈值化

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\binary.jpg)

##### 反二进制阈值化

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\binary_inv.jpg)

##### 阈值化为0

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\tozero.jpg)

##### 反阈值化为0

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\tozero_inv.jpg)

#####  截断阈值化

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\trunc.jpg)

##### 自适应阈值化

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_OSTU)
```

![binary](C:\Users\Administrator\OneDrive\笔记\opencv\ostu.jpg)

## 三、平滑处理

### 1. 均值滤波	cv2.blur()

```python
dst = cv2.blur(src, ksize)
```

#### 函数参数

- `src`：源图像
- `ksize`：卷积核大小，例如（N，N）

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
img = cv2.imread('lena.jpg')
dst1 = cv2.blur(img, (3,3))  # kernal 为3x3
dst2 = cv2.blur(img, (7,7))  # kernal 为7x7
result = np.hstack((img, dst1, dst2))
```

![blur](C:\Users\Administrator\OneDrive\笔记\opencv\blur.jpg)

### 2. 方框滤波	cv2.boxFilter()

```python
dst = cv2.boxFilter(src, ddepth, ksize, normalize=None)
```

#### 函数参数

- `src`：源图像
- `ddepth`：图像深度，一般取-1表示与源图像深度一致
- `ksize`：卷积核大小，例如（N，N）
- `normalize`：是否归一化，若为True，与均值滤波一致，若为False，大于255的值将被置为255

#### 返回值

- `dst`：处理结果

#### 实例效果

```
img = cv2.imread('lena.jpg')
dst1 = cv2.boxFilter(img, -1, (3,3), normalize=True)  # 进行归一化，与均值滤波相同
dst2 = cv2.boxFilter(img, -1, (3,3), normalize=False)  # 不进行归一化
result = np.hstack((img, dst1, dst2))
```

![boxFilter](C:\Users\Administrator\OneDrive\笔记\opencv\boxFilter.jpg)

### 3. 高斯滤波	cv2.GaussianBlur()

```python
dst = cv2.GaussianBlur(src, ksize, sigmaX)
```

#### 函数参数

- `src`：源图像
- `ksize`：卷积核大小，例如（N，N）
- `sigmaX`：X方向方差，控制卷积核权重

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
img = cv2.imread('lena.jpg')
dst1 = cv2.GaussianBlur(img, (3,3), 0)  # kernal 为3x3
dst2 = cv2.GaussianBlur(img, (7,7), 0)  # kernal 为7x7
result = np.hstack((img, dst1, dst2))
```

![gaussianBlur](C:\Users\Administrator\OneDrive\笔记\opencv\gaussianBlur.jpg)

### 4. 中值滤波	cv2.medianBlur()

```
dst = cv2.medianBlur(src, ksize)
```

#### 函数参数

- `src`：源图像
- `ksize`：卷积核大小，奇数整数值，例如3，5，7

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
img = cv2.imread('lena.jpg')
dst1 = cv2.medianBlur(img, 3)  # 核大小为3
dst2 = cv2.medianBlur(img, 7)  # 核大小为7
result = np.hstack((img, dst1, dst2))
```

![medianblur](C:\Users\Administrator\OneDrive\笔记\opencv\medianblur.jpg)

## 四、形态学操作

### 1. 腐蚀	cv2.erode(）

```python
dst = cv2.erode(src, kernel, iterations=None)
```

#### 函数参数

- `src`：源图像
- `kernal`：卷积核，常用`cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))`或`np.ones((5,5))`来获取
- `iterations`：进行腐蚀操作次数

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
img = cv2.imread('lena.jpg')
kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
dst = cv2.erode(img, kernal)
result = np.hstack((img, dst))
```

![](C:\Users\Administrator\OneDrive\笔记\opencv\erode.jpg)

### 2. 膨胀	cv2.dilate(）

```python
dst = cv2.dilate(src, kernel, iterations=None)
```

#### 函数参数

- `src`：源图像
- `kernal`：卷积核，常用`cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))`或`np.ones((5,5))`来获取
- `iterations`：进行腐蚀操作次数

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
dst = cv2.dilate(img, kernal)
result = np.hstack((img, dst))
```

![dilate](C:\Users\Administrator\OneDrive\笔记\opencv\dilate.jpg)

### 3. 组合	cv2.morphologyEx()

```python
dst = morphologyEx(src, op, kernel)
```

#### 函数参数

- `src`：源图像
- `op`：操作类别，常用：
  - `cv2.MORPH_CLOSE`：闭操作，先膨胀后腐蚀
  - `cv2.MORPH_OPEN`：开操作，先腐蚀后膨胀
  - `cv2.MORPH_TOPHAT`：顶貌操作，原图与开运算结果图之差
  - `cv2.MORPH_BLACKHAT`：黑帽操作，闭运算的结果与原图之差
  - `cv2.MORPH_GRADIENT`：形态梯度，膨胀与腐蚀之差
- `iterations`：进行腐蚀操作次数

#### 返回值

- `dst`：处理结果

#### 实例效果

##### 闭操作

先膨胀后腐蚀，目的是为了填充小的黑色区域

```python
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst1 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
```

![close](C:\Users\Administrator\OneDrive\笔记\opencv\close.jpg)

##### 开操作

先腐蚀后膨胀，目的是为了去除小的白色区域，去噪声去毛刺

```python
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst1 = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernal)
```

![open](C:\Users\Administrator\OneDrive\笔记\opencv\open.jpg)

##### 顶貌操作

原图与开运算结果图之差，用来分离比邻近点亮一些的斑块，得到噪声，得到亮的部分

```python
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dst1 = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernal)
dst2 = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernal)
```

![tophat](C:\Users\Administrator\OneDrive\笔记\opencv\tophat.jpg)

##### 黑帽操作

闭运算的结果与原图之差，用来分离比邻近点暗一些的斑，获取图像内部的小点，获暗的部分

```python
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
dst1 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
dst1 = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernal)
```

![blackhat](C:\Users\Administrator\OneDrive\笔记\opencv\blackhat.jpg)

##### 形态梯度

膨胀与腐蚀之差，获取边缘特征

```python
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dst1 = cv2.morphologyEx(src, cv2.cv2.MORPH_GRADIENT, kernal)
```

![gradient](C:\Users\Administrator\OneDrive\笔记\opencv\gradient.jpg)

## 五、梯度操作

### 1. Sobel算子	cv2.Sobel()

```
dst=cv2.Sobel(src, ddepth, dx, dy, ksize=None)
```

#### 函数参数

- `src`：源图像
- `ddepth`：图像深度，一般取-1表示与源图像深度一致。计算梯度值可能会出现负数，会自动截断为0，发生信息丢失。通常计算时，使用更高的数据类型`cv2.CV_64F`，再通过`cv2.convertScaleAbs(src)`取绝对值并转换为`cv2.CV_8U`类型。
- `dx`：计算x方向梯度，传入1或0
- `dy`：计算y方向梯度，传入1或0
- `ksize`：卷积核大小，整数，例如3、5、7

#### 返回值

- `dst`：处理结果

#### 实例效果

##### x方向梯度

```python
sobelx1 = cv2.Sobel(src, -1, 1, 0, ksize=3)  # 只计算黑到白的梯度
sobelx2 = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)  # 使用更高的数据类型`cv2.CV_64F
sobelx2 = cv2.convertScaleAbs(sobelx2)  #求绝对值获取双边梯度
```

![sobel_x](C:\Users\Administrator\OneDrive\笔记\opencv\sobel_x.jpg)

##### y方向梯度

```python
sobely1 = cv2.Sobel(src, -1, 0, 1, ksize=3)  # 只计算黑到白的梯度
sobely2 = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)  # 使用更高的数据类型`cv2.CV_64F
sobely2 = cv2.convertScaleAbs(sobely2)  #求绝对值获取双边梯度
```

![](C:\Users\Administrator\OneDrive\笔记\opencv\sobel_y.jpg)

##### 总梯度

* 方法一：分别计算x与y梯度然后相加

```python
sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
```

![](C:\Users\Administrator\OneDrive\笔记\opencv\sobel_xy.jpg)

* 方法二：直接计算总梯度

```python
sobelxy = cv2.Sobel(src, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
```

![sobel_xy2](C:\Users\Administrator\OneDrive\笔记\opencv\sobel_xy2.jpg)

### 2. Scharr算子	cv2.Scharr()

```
dst=cv2.Scharr(src, ddepth, dx, dy)
```

#### 函数参数

- `src`：源图像
- `ddepth`：图像深度，一般取-1表示与源图像深度一致。计算梯度值可能会出现负数，会自动截断为0，发生信息丢失。通常计算时，使用更高的数据类型`cv2.CV_64F`，再通过`cv2.convertScaleAbs(src)`取绝对值并转换为`cv2.CV_8U`类型。
- `dx`：计算x方向梯度，传入1或0
- `dy`：计算y方向梯度，传入1或0

#### 返回值

- `dst`：处理结果

#### 实例效果

##### x方向梯度

```python
scharrx1 = cv2.Scharr(src, -1, 1, 0)  # 只计算黑到白的梯度
scharrx2 = cv2.Scharr(src, cv2.CV_64F, 1, 0)  # 使用更高的数据类型`cv2.CV_64F
scharrx2 = cv2.convertScaleAbs(scharrx2)  #求绝对值获取双边梯度
```

![sobel_x](C:\Users\Administrator\OneDrive\笔记\opencv\scharr_x.jpg)

##### y方向梯度

```python
scharry1 = cv2.Scharr(src, -1, 0, 1)  # 只计算黑到白的梯度
scharry2 = cv2.Scharr(src, cv2.CV_64F, 0, 1)  # 使用更高的数据类型`cv2.CV_64F
scharry2 = cv2.convertScaleAbs(scharry2)  #求绝对值获取双边梯度
```

![](C:\Users\Administrator\OneDrive\笔记\opencv\scharr_y.jpg)

##### 总梯度

```python
scharrx = cv2.Scharr(src, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(src, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
```

![scharr_xy](C:\Users\Administrator\OneDrive\笔记\opencv\scharr_xy.jpg)

### 3. Laplacian算子	cv2. Laplacian()

```python
dst=cv2.Laplacian(src, ddepth)
```

#### 函数参数

- `src`：源图像
- `ddepth`：图像深度，一般取-1表示与源图像深度一致。计算梯度值可能会出现负数，会自动截断为0，发生信息丢失。通常计算时，使用更高的数据类型`cv2.CV_64F`，再通过`cv2.convertScaleAbs(src)`取绝对值并转换为`cv2.CV_8U`类型。

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
laplacian = cv2.Laplacian(src, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
```

![](C:\Users\Administrator\OneDrive\笔记\opencv\laplacian_xy.jpg)

### 4. 各算子对比

从左到右依次为：原图像、Sobel算子、Scharr算子、Laplacian算子

![compare](C:\Users\Administrator\OneDrive\笔记\opencv\compare.jpg)

## 六、边缘检测

###  1. Canny边缘检测	cv2.Canny()

原理：高斯滤波去噪+梯度大小与方向+非极大值抑制+滞后阈值

```python
dst = Canny(src, threshold1, threshold2)
```

#### 函数参数

- `src`：源图像

- `threshold1`：滞后阈值的低阈值

- `threshold2`：滞后阈值的高阈值

  ![滞后阈值](C:\Users\Administrator\OneDrive\笔记\opencv\滞后阈值.png)

#### 返回值

- `dst`：处理结果

#### 实例效果

```python
thred1 = [50, 100, 150]
thred2 = [50, 100, 150]
for t1 in thred1:    
	for t2 in thred2:        
		dst = cv2.Canny(src, t1, t2)
```

![canny_lena](C:\Users\Administrator\OneDrive\笔记\opencv\canny_lena.jpg)

