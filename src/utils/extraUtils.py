import pandas as pd
import matplotlib.pyplot as plt

def load_csv(file_path):
    data = pd.read_csv(file_path)
    loss = data['loss'].values
    accuracy = data['accuracy'].values
    val_loss = data['val_loss'].values
    val_accuracy = data['val_accuracy'].values
    return loss, accuracy, val_loss, val_accuracy

def load_all_csv(file_path):
    data = pd.read_csv(file_path)
    loss = data['loss'].values
    accuracy = data['accuracy'].values
    val_loss = data['val_loss'].values
    val_accuracy = data['val_accuracy'].values
    return loss, accuracy, val_loss, val_accuracy

def plot_graphs(loss, accuracy, val_loss, val_accuracy):
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss', fontsize='large')
    plt.xlabel('Epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('epochs_loss.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy', fontsize='large')
    plt.xlabel('Epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('epochs_accuracy.png', bbox_inches='tight')
    plt.close()

# List of file paths for the CSV files
csv_files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv', 'file7.csv', 'file8.csv', 'file9.csv', 'file10.csv']

loss_data = []
accuracy_data = []
val_loss_data = []
val_accuracy_data = []

# Load and collect data from each CSV file
for file_path in csv_files:
    loss, accuracy, val_loss, val_accuracy = load_csv(file_path)
    loss_data.append(loss)
    accuracy_data.append(accuracy)
    val_loss_data.append(val_loss)
    val_accuracy_data.append(val_accuracy)

# Plot the graphs overlaying all the CSVs
plot_graphs(loss_data, accuracy_data, val_loss_data, val_accuracy_data)
