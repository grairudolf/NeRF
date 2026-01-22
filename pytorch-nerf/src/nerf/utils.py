def log_message(message):
    print(f"[INFO] {message}")

def visualize_images(images, titles=None, cols=3):
    import matplotlib.pyplot as plt
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i])
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_checkpoint(state, filename='checkpoint.pth'):
    import torch
    torch.save(state, filename)
    log_message(f"Checkpoint saved to {filename}")

def load_checkpoint(filename='checkpoint.pth'):
    import torch
    checkpoint = torch.load(filename)
    log_message(f"Checkpoint loaded from {filename}")
    return checkpoint

def create_directory(dir_path):
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        log_message(f"Directory created: {dir_path}")
    else:
        log_message(f"Directory already exists: {dir_path}")