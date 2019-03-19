# ParallelSMO

*本篇论文已发表在HPCC-2018上（oral），针对SMO算法进行了改进，提出了并行的思路和实现（IPSMO-1 AND IPSMO-2）。*

## SMO
N代表了样本的个数，当样本量小时，目标函数可以直接解出；当N很大时，二次规划问题的复杂度将会上升，直接求解将会变得很困难。
于是有了SMO。

序列最小最优化算法(SMO)可以高效的求解SVM对偶问题，它把原始求解N个参数二次规划问题分解成很多个子二次规划问题分别求解，每个子问题只需要求解2个参数，方法类似于坐标上升，节省时间成本和降低了内存需求。每次启发式选择两个变量进行优化，不断循环，直到达到函数最优值。并且alpha的更新规则比求解对偶问题更简单。

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/svm%E5%AF%B9%E5%81%B6%E9%97%AE%E9%A2%98.png" width="400" alt="SVM对偶问题">

《统计学习方法》
SMO的基本思路：如果所有变量的解都满足此优化问题的KKT条件，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。否则，选择两个变量，固定其他变量，针对这两个变量构造一个二次规划问题，这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。子问题有两个变量，一个是违反KKT条件最严重的那一个，另一个由约束条件自动确定。如此，SMO算法将原问题不断分解为子问题，并对子问题求解，进而达到求解原问题的目的。

SMO算法将原问题不断分解为子问题，那么我们什么时候停止对原问题的分解呢？这里需要了解KKT条件，因为满足了KKT条件，最优解就找到了，就可以停止了。

由此推导出了最优化终止条件：这里和《西瓜书》和《统计学习方法》上略有不同，具体来源可以参考论文呢的参考文献。

底下图片中推导的工作集选择算法，并不是libsvm中所用的。（具体请看论文！）

## IPSMO-1 & IPSMO-2

考虑到用latex写完这么多公式，怕是要等到我找到正式工作了

所以。。。看图片

还有论文是修改的libsvm的源码，实验数据来自于libsvm的官网。

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/949241553000557_.pic_hd.jpg" width="1200" alt="note1">

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/949251553000558_.pic_hd.jpg" width="1200" alt="note2">

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/949261553000559_.pic_hd.jpg" width="1200" alt="note3">

## Experiment Results

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/ex-1.png" width="1200" alt="note3">

<img src="https://github.com/michelleweii/ParallelSMO/blob/master/picture/ex-2.png" width="1200" alt="note3">

放了两张表，实验结果还是不错的。
