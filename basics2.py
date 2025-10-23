import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Lists to store metrics for plotting
loss_values = []
accuracy_values = []
f1_scores = []

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale and normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 1000
learning_rate = 0.04
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()
    loss_values.append(loss.item())
    
    with torch.no_grad():
        y_predicted1 = model(X_train)
        y_predicted_cls1 = (y_predicted1 >= 0.6).float()
        acc = y_predicted_cls1.eq(y_train).sum() / float(y_train.shape[0])
        accuracy_values.append(acc.item())

        # Calculate F1-score
        f1 = f1_score(y_train.numpy(), y_predicted_cls1.numpy())
        f1_scores.append(f1)

    if (epoch+1) % 2000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# Extract weights and bias for the final linear equation
[w, b] = model.parameters()
w = w.detach().numpy().flatten()  # Convert weights to numpy array
b = b.item()  # Convert bias to scalar

# Display the final linear equation
equation = "y = " + " + ".join([f"{w[i]:.4f}*x{i+1}" for i in range(len(w))]) + f" + {b:.4f}"
print("Final Linear Equation:", equation)


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = (y_predicted>=0.6).float()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f' final accuracy on test data: {acc.item():.4f}')



# Plot loss, accuracy, and F1-score in one figure
plt.figure(figsize=(10, 15))

# Plot Loss
plt.subplot(3, 1, 1)
plt.plot(range(1, num_epochs + 1), loss_values, label='Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(3, 1, 2)
plt.plot(range(1, num_epochs + 1), accuracy_values, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid()

# Plot F1-Score
plt.subplot(3, 1, 3)
plt.plot(range(1, num_epochs + 1), f1_scores, label='F1-Score', color='red')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score Over Epochs')
plt.legend()
plt.grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



# with torch.no_grad():
#     y_predicted = model(X_test).squeeze().numpy()  # Predicted probabilities
#     y_true = y_test.squeeze().numpy()              # True labels

# threshold = 0.6

# # Assign colors and markers for each case
# colors = []
# markers = []
# for yt, yp in zip(y_true, y_predicted):
#     pred = int(yp >= threshold)
#     if pred == 1 and yt == 1:
#         colors.append('purple')   # True Positive
#         markers.append('s')
#     elif pred == 1 and yt == 0:
#         colors.append('gold')     # False Positive
#         markers.append('X')
#     elif pred == 0 and yt == 0:
#         colors.append('gold')     # True Negative
#         markers.append('o')
#     elif pred == 0 and yt == 1:
#         colors.append('purple')   # False Negative
#         markers.append('X')

# # Plot
# plt.figure(figsize=(12, 4))
# for i, (yp, c, m) in enumerate(zip(y_predicted, colors, markers)):
#     plt.scatter(yp, 0, color=c, marker=m, s=100, edgecolor='k', alpha=0.7)

# plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
# plt.yticks([])
# plt.xlabel('Predicted Probability')
# plt.title('Classification Threshold Visualization')
# plt.legend(['Threshold'])
# plt.tight_layout()
# plt.show()
