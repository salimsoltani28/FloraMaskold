import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import load_config
from dataset import build_loader
from models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
config_file = '/mnt/gsdata/users/soltani/Workshop_home_fromSSD2/Workshop_home/4_Mask2Former/2_MyDiv/configs/finetune_maskformer2.yml'
config = load_config(config_file)

# Build data loaders
train_subset, val_subset, train_loader, val_loader = build_loader(config)

# Define the model
model = build_model(config['model'])
model.to(device)

# Check the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params / 1e6:.2f} million")

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['base_lr'], eps=config['train']['optimizer']['eps'], betas=tuple(config['train']['optimizer']['betas']))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['lr_scheduler']['decay_epochs'], gamma=config['train']['lr_scheduler']['cycle_limit'])

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training settings
num_epochs = config['train']['epochs']
checkpoint_path = "/mnt/gsdata/users/soltani/Workshop_home_fromSSD2/Workshop_home/4_Mask2Former/2_MyDiv/checkpoints/best_model_checkpoint2.pth"

# Training loop
best_loss = float('inf')
best_epoch = -1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(batch, device)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch['pixel_values'].size(0)
        num_samples += batch['pixel_values'].size(0)

    epoch_loss = running_loss / num_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            with torch.cuda.amp.autocast():
                outputs = model(batch, device)
                loss = outputs.loss

            val_loss += loss.item() * batch['pixel_values'].size(0)
            val_samples += batch['pixel_values'].size(0)

    avg_val_loss = val_loss / val_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
    # Update the learning rate
    scheduler.step()

    # Save the best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, checkpoint_path)
        print(f"Saved best model at epoch {epoch+1} with validation loss {best_loss:.4f}")

print(f"Training completed. Best model saved at epoch {best_epoch+1} with validation loss {best_loss:.4f}")
