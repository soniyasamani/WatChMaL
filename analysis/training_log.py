import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
model='FirstRed'
path='/opt/ppd/hyperk/Users/samanis/WatChMaL/outputs/resnet18/firstred_ndet/'
fig_name='/opt/ppd/hyperk/Users/samanis/WatChMaL/analysis/figures/training/%s.png'%(model)
data = pd.read_csv(path+'log_train_0.csv')
val_data = pd.read_csv(path+'log_val.csv')

# Convert the epochs to integers
data['epoch'] = data['epoch'].astype(int)
val_data['epoch'] = val_data['epoch'].astype(int)

# Extract necessary data
epochs = data['epoch']
loss = data['loss']
accuracy = data['accuracy']

# Extract validation data
val_epochs = val_data['epoch']
val_loss = val_data['loss']
val_accuracy = val_data['accuracy']

# Create continuous x-axis values for better positioning
epochs = np.linspace(epochs.min(), epochs.max(), len(epochs))
val_epochs = np.linspace(val_epochs.min(), val_epochs.max(), len(val_epochs))

# Find the epoch with the lowest validation loss
min_val_loss_epoch = val_epochs[val_loss.idxmin()]
min_val_loss = val_loss.min()
max_val_accuracy = val_accuracy[val_loss.idxmin()]  # Get accuracy at the same index

# Create the plot
fig, ax1 = plt.subplots(figsize=(15, 8))
# Create a second y-axis
ax2 = ax1.twinx()
# Set the background color of the plot area
ax1.set_facecolor((0.95, 0.95, 0.95))  # RGB values range from 0 to 1
ax2.set_facecolor((0.95, 0.95, 0.95))  # RGB values range from 0 to 1

# Plot loss in blue with solid line
ax1.plot(epochs, loss, 'b-', label='Loss', linewidth=2, alpha=0.5)
ax1.scatter(val_epochs, val_loss, color='b', marker='o', label = "Val Loss")
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Loss', color='b', fontsize=18)
ax1.tick_params(axis='y', which='major', labelsize=14, colors='b')
ax2.spines['left'].set_color('b')        # setting up Y-axis tick color to blue

# Plot accuracy in red with solid line
ax2.plot(epochs, accuracy, 'r-', label='Accuracy', linewidth=2, alpha=0.5)
ax2.scatter(val_epochs, val_accuracy, color='r', marker='o',label="Val Accuracy")
ax2.set_ylabel('Accuracy', color='r', fontsize=18)
ax2.tick_params(axis='y', which='major', labelsize=14, colors='r')
ax2.spines['right'].set_color('r')

# Add a vertical black line at the point of minimum validation loss
ax1.axvline(x=min_val_loss_epoch, color='black', linestyle='--',linewidth=2)
# Add markers "X" at the point of minimum validation loss and max accuracy
ax1.scatter(min_val_loss_epoch, min_val_loss, color='darkblue', marker='x', s=500, linewidth=4)
ax1.scatter(min_val_loss_epoch, max_val_accuracy, color='darkred', marker='x', s=500, linewidth=4)

# Add vertical text along the black dashed line (centered)
ax1.text(min_val_loss_epoch, (min_val_loss + max_val_accuracy) / 2, 'Model Saved', rotation=90, va='center', ha='right', fontsize=20, color='black')
# Add a gray shaded region for everything after the vertical black dashed line
ax1.axvspan(min_val_loss_epoch, max(epochs), facecolor='gray', alpha=0.5)
# Add text for minimum loss and maximum accuracy values
ax1.text(min_val_loss_epoch+0.11, min_val_loss-0.02, f'Min Loss: {min_val_loss:.4f}', va='bottom', ha='left', fontsize=18, color='darkblue',zorder=4)
ax1.text(min_val_loss_epoch+0.11, max_val_accuracy-0.02, f'Max Acc: {max_val_accuracy:.4f}', va='bottom', ha='left', fontsize=18, color='darkred',zorder=4)

# Set x-axis limits from 0 to 20 and set ticks at integer positions
ax1.set_xlim(0, max(epochs))
ax1.set_ylim(0, 1.0)
ax2.set_ylim(0,1.0)
plt.xticks(range(int(max(epochs))+1))

# Add grid lines
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add title
plt.title('Loss and Accuracy over Epochs (%s)'%(model), fontsize=16)

# Add legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left', ncol=2, fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

fig.savefig(fig_name, dpi=1200, bbox_inches='tight', format='png')