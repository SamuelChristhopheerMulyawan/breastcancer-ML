import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

dataset = pd.read_csv("dataset.csv")
test = pd.read_csv("test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
test_data = test.iloc[:, 2:32].values
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
data = dataset.iloc[:, 2:32].values
label = dataset.iloc[:, 1].values
dataTensor = torch.tensor(data, dtype=torch.float32).to(device)
labelTensor = torch.tensor(label, dtype=torch.float32).to(device)

DataSet = TensorDataset(dataTensor, labelTensor)
Loader = DataLoader(DataSet, batch_size=16, shuffle=False)
model = nn.Sequential(
    nn.Linear(30, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
).to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch_x, batch_y in Loader:
        output = model(batch_x).squeeze()
        loss = loss_fn(output, batch_y).squeeze()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 :
        print(f"Epoch {epoch}, Loss {total_loss / len(Loader):.4f}")

model.eval()
with torch.no_grad():
    testTensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    sample = testTensor[7] # the 7th data on test.csv
    prob = model(sample).item()
    pred = 1 if prob >= 0.5 else 0
    print(f"{'Cancer' if pred else 'Not Cancer'}")

