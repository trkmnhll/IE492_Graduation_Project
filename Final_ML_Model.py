import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import matplotlib.pyplot as plt

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def load_data_from_csv(filename):

    inputs, targets, heuristics = [], [], []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

        # Process the CSV rows in triplets: one for inputs, one for targets, one for heuristic sequence
        for i in range(0, len(data), 3):
            try:
                # Parse the input matrix (row i)
                input_matrix = eval(data[i][0])  # Convert the string to a Python list
                input_array = np.array(input_matrix, dtype=np.float32)  # Convert to NumPy array
                inputs.append(input_array)

                # Parse the target list (row i + 1)
                target_list = eval(data[i + 1][0])  # Convert the string to a Python list
                target_array = np.array(target_list, dtype=np.float32)  # Convert to NumPy array
                targets.append(target_array)

                # Parse the heuristic sequence (row i + 2)
                heuristic_sequence = eval(data[i + 2][0])  # Convert the string to a Python list
                heuristics.append(heuristic_sequence)
            except Exception as e:
                print(f"Error processing rows {i}, {i + 1}, and {i + 2}: {e}")
                continue

    return inputs, targets, heuristics
# Replace with your CSV file path
filename = 'loose_data.csv'

# Load data
inputs, targets, heuristics = load_data_from_csv(filename)

# Convert data to tensors
# Convert data to NumPy arrays first
inputs = np.stack(inputs)  # Stack into a single NumPy array
targets = np.stack(targets)  # Stack into a single NumPy array
heuristics = np.stack(heuristics)  # Stack into a single NumPy array

# Convert to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
heuristics = torch.tensor(heuristics, dtype=torch.float32)

train_inputs, test_inputs, train_targets, test_targets, train_heuristics, test_heuristics = train_test_split(
    inputs, targets, heuristics, test_size=0.2, random_state=42)

# Define the RankNet model
class RankNet(nn.Module):
    def _init_(self):
        super(RankNet, self)._init_()
        self.hidden1 = nn.Linear(50 * 3, 150)
        self.hidden2 = nn.Linear(150,100)
        self.hidden3 = nn.Linear(100,50)
        self.output = nn.Linear(50, 50)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.output(x)  # No softmax; handled in loss
        return x

# Initialize
model = RankNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.apply(init_weights)
# Convert targets for CrossEntropyLoss
train_targets = torch.argmax(train_targets, dim=-1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Initialize variables for evaluation
model.eval()
total_prediction_order = 0
predicted_jobs = []
correct_predictions = 0

# Dictionary to store counts of prediction orders
order_counts = {}

print("Predicted Jobs:")
with torch.no_grad():
    for idx, (test_input, heuristic) in enumerate(zip(test_inputs, test_heuristics)):
        test_input = test_input.unsqueeze(0)  # Add batch dimension
        test_output = model(test_input)
        predicted_job = test_output.argmax(dim=-1).item() + 1  # Add 1 to convert index to job number
        
        # Find the order of the predicted job in the heuristic
        prediction_order = (heuristic == predicted_job).nonzero(as_tuple=True)[0].item() + 1
        total_prediction_order += prediction_order

        # Update the order counts
        if prediction_order in order_counts:
            order_counts[prediction_order] += 1
        else:
            order_counts[prediction_order] = 1

        # Check if prediction matches the first job in heuristic
        actual_first_job = heuristic[0].item()  # Heuristic's first job
        if predicted_job == actual_first_job:
            correct_predictions += 1
        
        # Save predicted job for display
        predicted_jobs.append(predicted_job)

        # Print for each test input
        print(f"Test Sample {idx + 1}: Predicted Job = {predicted_job} Order = {prediction_order} Heuristic = {heuristic.tolist()}")

# Calculate average prediction order
average_prediction_order = total_prediction_order / len(test_inputs)
print(f"\nAverage Prediction Order: {average_prediction_order:.4f}")
test_accuracy = correct_predictions / len(test_inputs) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Display prediction order counts
print("\nPrediction Order Counts:")
for order, count in sorted(order_counts.items()):
    print(f"Order {order}: {count} times")

# Optional: Summarize total predictions
total_predictions = sum(order_counts.values())
print(f"\nTotal Predictions: {total_predictions}")

def plot_prediction_order_counts(order_counts):
    plt.figure(figsize=(10, 6))  # Set figure size

    # Extract the sorted orders and corresponding counts
    orders = sorted(order_counts.keys())  # Sorted order numbers (x-axis)
    counts = [order_counts[order] for order in orders]  # Counts for each order (y-axis)

    # Plot the bar chart
    plt.bar(orders, counts, color='skyblue')

    # Add labels and title
    plt.xlabel('Prediction Order', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Prediction Order Distribution', fontsize=16)

    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(orders[i], count + 0.01 * max(counts), str(count), ha='center', va='bottom', fontsize=10)

    # Customize the x-axis to ensure all orders are displayed
    plt.xticks(orders, fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('loose_prediction_order_bar_chart.png')  # Save as an image file
    plt.show()

# Call the function to plot the bar chart
plot_prediction_order_counts(order_counts)