# Deep Learning Image Classification Models Based Vision-Attention
this is a pytorch deep learning project for the five flowers' classification project with ViT(vision_transformer).

这是一个基于ViT的花朵图像分类的pytorch深度学习项目。

# environment
python:3.8

pytorch:11.8

IDE:pycharm

# application
The train.py script trains the data using the ViT model, records and outputs the loss, train accuracy, and test accuracy for each epoch, and generates a .pth file to store the trained model weights. The test.py script loads the model weights from the .pth file and predicts the image data, with the option to set the img_path parameter to specify the image to be tested. Finally, it generates the probabilities of the image belonging to each category.

train.py运行会使用ViT模型对数据进行训练，并记录并输出每一个epoch的loss，train acc和test acc，并生成.pth文件存储模型训练权重；test.py文件会加载.pth文件中的模型权重，并对图片数据进行预测，可用img_path参数设置要训练的图片，最后生成该图片是各个种类的可能性。
