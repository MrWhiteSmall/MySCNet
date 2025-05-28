# MySCNet

1. 一个目的是 统一其中使用的四个ReID数据集的加载方式，全都使用vcclothes的文件命名,操作文件写在 unify_data 中 , 最终文件名都是 humanid_cameraid_clothid_xx.jpg/png，可以直接使用vcclothes.py的dataloader进行加载
2. 一个目的是 训练后，编写 单个 人物预测的脚本，输出其pred_id accuracy，可以用作其他方面的使用（这里将会作为我另一个工作的reward函数，计算经过干扰以后的human是否会产生预测的偏差，出现偏差则给奖励，奖励的计算方式 =  (id==pred_id ? 0 : 1) + (accuracy-mean_acc)/std_acc ）
