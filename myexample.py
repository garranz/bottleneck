from mymodels import DIBnet
import torch as tc
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split

# Example custom dataset
class MyDataset(Dataset):
    def __init__(self, input1, input2, output):
        self.input1 = input1
        self.input2 = input2
        self.output = output

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        return self.input1[idx], self.input2[idx], self.output[idx]


if __name__ == "__main__":

    N = 100_000

    m1, m2 = 0, 2
    s1, s2 = 5, 8

    x1 = tc.randn( N,1 )*s1 + m1 
    x2 = tc.randn( N,1 )*s2 + m2

    y  = x1**2 + x1*x2

    model = DIBnet( (1,1), 1, in_emb_d=1, Earch=(64,64), Oarch=(64,32) )

    optimizer = tc.optim.Adam( model.parameters(), lr=1e-3 ) # pyright:ignore

    device = 'mps'
    model.to(device)

    x1 = (x1 - x1.mean())/x1.std()
    x2 = (x2 - x2.mean())/x2.std()
    y = (y - y.mean())/y.std()
    '''
    x1c = x1.to(device)
    x2c = x2.to(device)
    yc = y.to(device)


    # Creating the dataset
    dataset = MyDataset(x1c, x2c, yc)

    # Splitting the dataset (70-30 split for training and validation)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)


    for bet in (.001,):# .01, .1, 1.):

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):

            model.train()  # Set model to training mode
            running_loss = 0.0

            # Example loop over the DataLoader
            for ib, (x1b, x2b, yb) in enumerate(train_loader):
    
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                Lmi, KLs, klarray = model( (x1b, x2b), yb )

                # Compute loss
                loss = -Lmi #+ bet*KLs

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                print( f'Lmi: {Lmi}', f'KLsum = {KLs}, KLs:{klarray}' )
                # Accumulate loss for reporting
                running_loss += loss.item()

            # Print average loss for the epoch
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

            # Validation step (optional)
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with tc.no_grad():
                for x1b, x2b, yb in val_loader:
                    Lmi, KLs, _ = model( (x1b, x2b), yb )
                    val_loss += -Lmi #+ bet*KLs

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss}")
    '''
