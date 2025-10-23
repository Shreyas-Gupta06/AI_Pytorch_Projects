# import numpy as np 

# # Compute every step manually

# # Linear regression
# # f = w * x 

# # here : f = 2 * x
# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)

# w = 0.0

# # model output
# def forward(x):
#     return w * x

# # loss = MSE
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()

# # J = MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N * 2x(w*x - y)
# def gradient(x, y, y_pred):
#     # print(f'Gradient: {x}, {y}, {y_pred}')
#     # print(2*x*(y_pred - y))
#     # print((2*x*(y_pred - y)).mean())
#     return (2*x*(y_pred - y)).mean()


# print(f'Prediction before training: f(5) = {forward(5):.3f}')

# # Training
# learning_rate = 0.01
# n_iters = 10

# for epoch in range(n_iters):
#     # predict = forward pass
#     y_pred = forward(X)

#     # loss
#     l = loss(Y, y_pred)
    
#     # calculate gradients
#     dw = gradient(X, Y, y_pred)

#     # update weights
#     w -= learning_rate * dw

#     if epoch % 1 == 0:
#         print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
# print(f'Prediction after training: f(5) = {forward(5):.3f}')

# # /////////////////////////////////////// use autograd for gradient computation

# import torch

# # Here we replace the manually computed gradient with autograd

# # Linear regression
# # f = w * x 

# # here : f = 2 * x
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# # model output
# def forward(x):
#     return w * x

# # loss = MSE
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()

# print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# # Training
# learning_rate = 0.01
# n_iters = 100

# for epoch in range(n_iters):
#     # predict = forward pass
#     y_pred = forward(X)

#     # loss
#     l = loss(Y, y_pred)

#     # calculate gradients = backward pass
#     l.backward()

#     # update weights
#     #w.data = w.data - learning_rate * w.grad
#     with torch.no_grad():
#         w -= learning_rate * w.grad
    
#     # zero the gradients after updating
#     w.grad.zero_()

#     if epoch % 10 == 0:
#         print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

# print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

# # /////////////////////////////////////// use pytorch for loss computation, parameters updates and autograd

# # 1) Design model (input, output, forward pass with different layers)
# # 2) Construct loss and optimizer
# # 3) Training loop
# #       - Forward = compute prediction and loss
# #       - Backward = compute gradients
# #       - Update weights

# import torch
# import torch.nn as nn

# # Linear regression
# # f = w * x 

# # here : f = 2 * x

# # 0) Training samples, watch the shape!
# X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
# Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# n_samples, n_features = X.shape
# print(f'#samples: {n_samples}, #features: {n_features}')
# # 0) create a test sample
# X_test = torch.tensor([5], dtype=torch.float32)
# print(X_test.shape)

# # 1) Design Model, the model has to implement the forward pass!
# # Here we can use a built-in model from PyTorch
# input_size = n_features
# output_size = n_features

# # we can call this model with samples X
# model = nn.Linear(input_size, output_size)

# '''
# class LinearRegression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         # define diferent layers
#         self.lin = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.lin(x)

# model = LinearRegression(input_size, output_size)
# '''

# print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# # 2) Define loss and optimizer
# learning_rate = 0.01
# n_iters = 100

# loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # 3) Training loop
# for epoch in range(n_iters):
#     # predict = forward pass with our model
#     y_predicted = model(X)

#     # loss
#     l = loss(Y, y_predicted)

#     # calculate gradients = backward pass
#     l.backward()

#     # update weights
#     optimizer.step()

#     # zero the gradients after updating
#     optimizer.zero_grad()

#     if epoch % 10 == 0:
#         [w, b] = model.parameters() # unpack parameters
#         print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

# print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')



# /////////////////////////////////////// custom example with 2 features and plotting



import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Lists to store values for plotting
loss_values = []
w_values = []
b_values = []



X = torch.tensor([[1, 2],  # Sample 1: x1=1, x2=2
                  [3, 4],  # Sample 2: x1=3, x2=4
                  [5, 6],  # Sample 3: x1=5, x2=6
                  [7, 8]], # Sample 4: x1=7, x2=8
                 dtype=torch.float32)

# Calculate Y using the equation y = 2x1 + x2 - 1
Y = torch.tensor([[2*1 + 2 - 1],  # Target for Sample 1
                  [2*3 + 4 - 1],  # Target for Sample 2
                  [2*5 + 6 - 1],  # Target for Sample 3
                  [2*7 + 8 - 1]], # Target for Sample 4  
                 dtype=torch.float32)



X_test = torch.tensor([[0,1]], dtype=torch.float32)
print(X_test.shape)
# x.samples = 4, x.features = 2, output size = 1(one output for each input), input size = 2(x1,x2) = x_features

n_samples, n_features = X.shape
input_size = n_features # 2
output_size = 1

# we can call this model with samples X
model = nn.Linear(input_size, output_size) # weights have shape output_size x input_size = 1 x 2, bias has shape output_size = 1


print(f'Prediction before training: f(0,1) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

c = 50

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    # Store values every c epochs
    if epoch % c == 0:
        [w, b] = model.parameters()  # Unpack weights and bias
        loss_values.append(l.item())  # Store loss
        w_values.append(w.detach().numpy().flatten())  # Store weights
        b_values.append(b.item())  # Store bias




    if epoch % 100 == 0:
        [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}: w = {w}, b = {b}, loss = {l.item():.4f} \n')


# Extract weights and bias after training (plotting linear regression line)
[w, b] = model.parameters()
w = w.detach().numpy().flatten()  # Convert weights to numpy array
b = b.item()  # Convert bias to scalar

# Generate points for the regression line
x1_range = torch.linspace(0, 10, 100)  # Generate 100 points for x1
x2_fixed = 5  # Fix x2 to a constant value (e.g., 5)
y_line = w[0] * x1_range + w[1] * x2_fixed + b  # Compute y = w1*x1 + w2*x2 + b

# Plot data points
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], Y, color='blue', label='x1 points')  # x1 points
plt.scatter(X[:, 1], Y, color='green', label='x2 points')  # x2 points

# Plot regression line
plt.plot(x1_range, y_line, color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('Feature Values')
plt.ylabel('Target Values')
plt.title('Linear Regression: Data Points and Regression Line')
plt.legend()
plt.grid()
plt.show()




# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(range(0,n_iters,c), loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Plot Weights
w_values = list(zip(*w_values))  # Transpose the list of weights
plt.figure(figsize=(10, 6))
for i, w in enumerate(w_values):
    plt.plot(range(0,n_iters,c), w, label=f'w{i+1}')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('Weights Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Plot Bias
plt.figure(figsize=(10, 6))
plt.plot(range(0,n_iters,c), b_values, label='Bias', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Bias Value')
plt.title('Bias Over Epochs')
plt.legend()
plt.grid()
plt.show()




print(f'Prediction after training: f(0,1) = {model(X_test).item():.3f}')


