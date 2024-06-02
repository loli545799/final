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
    plt.savefig('output.png')  # 保存图像到当前目录下的 output.png 文件
    
