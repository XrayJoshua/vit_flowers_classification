import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from transformer.vision_transformer import vit_base_patch32_224


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_path = 'D:\\PycharmProjects\\pythonProject\\vit_flowers_classification\\Datasets\\flower'
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        "val": transforms.Compose([transforms.Resize((224, 224)),  # val不需要任何数据增强
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(data_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = vit_base_patch32_224(num_classes=5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    epochs = 5
    save_path = os.path.abspath(os.path.join(os.getcwd(), './results/weights/transformer'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_acc = 0.0
    for epoch in range(epochs):
        ############################################################## train ######################################################
        net.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for data in train_bar:
            images, labels = data
            sample_num += images.shape[0]
            optimizer.zero_grad()
            outputs = net(images.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(pred_class, labels.to(device)).sum()
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num.item() / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc_num = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc_num / val_num
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
        epoch + 1, loss, train_acc, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_path, "Transformer.pth"))

        train_acc = 0.0
        val_accurate = 0.0

    print('Finished Training')


main()
