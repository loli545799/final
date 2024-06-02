import matplotlib.pyplot as plt
def show_data(train_losses,val_losses,psnrs,ssims):
    
    epochs = range(1, len(train_losses) + 1)

    # 设置图形窗口
    plt.figure(figsize=(15, 5))  # 宽度 15 英寸，高度 5 英寸

    # 第一个子图：绘制损失
    plt.subplot(1, 3, 1)  # 1 行 3 列的第一个
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 第二个子图：绘制 PSNR
    plt.subplot(1, 3, 2)  # 1 行 3 列的第二个
    plt.plot(epochs, psnrs, label='PSNR', color='green', marker='s')
    plt.title('PSNR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.ylim(0, 20.0) 
    plt.legend()

    # 第三个子图：绘制 SSIM
    plt.subplot(1, 3, 3)  # 1 行 3 列的第三个
    plt.plot(epochs, ssims, label='SSIM', color='red', marker='^')
    plt.title('SSIM over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.ylim(0, max(ssims) * 1.1) 
    plt.legend()

    # 显示整个图形窗口
    plt.tight_layout()  # 调整子图间距
    plt.savefig('data_view.png')  # 保存图像到当前目录下的 output.png 文件

train_losses = [0.1359, 0.0184, 0.0185, 0.0166, 0.0164, 0.0162, 0.0163, 0.0161, 0.0161, 0.0161]
val_losses = [0.0195, 0.0161, 0.0182, 0.0163, 0.0160, 0.0158, 0.0185, 0.0174, 0.0157, 0.0157]

psnrs = [17.1489, 18.0128, 17.5135, 17.9720, 18.0325, 18.0626, 17.3654, 17.6493, 18.1329, 18.1346]

# 假设的 SSIM 数据
ssims = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88]
show_data(train_losses,val_losses,psnrs,ssims)
