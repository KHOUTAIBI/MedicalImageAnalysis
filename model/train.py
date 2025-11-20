from model.spherical_vae import *
from dataloader.utils import * 
from tqdm import tqdm

# -------------------------- Training Loop ----------------------------

def train(model, train_loader, test_loader, optimizer, scheduler):
    """
    Training the model
    """
    # losses
    n_epochs = model.config["n_epochs"]
    
    for epoch in tqdm(range(n_epochs), desc='Epoch'):

        batch_train_tqdm = tqdm(train_loader, desc='Batch Train', leave=False)
        batch_test_tqdm = tqdm(test_loader, desc='Batch Loss', leave=False)

        train_loss = 0
        test_loss = 0
        epoch_train_losses = []
        epoch_test_losses = []

        model.train()
        
        for batch_data in batch_train_tqdm :

            # Looping over and calculating stuff
            data, labels = batch_data
            data = data.to(model.device)
            labels = labels.to(model.device)
            optimizer.zero_grad()
            z_batch, x_mu_batch, posterior_params = model(data)

            elbo_loss = model._elbo(data, x_mu_batch, posterior_params)

            # loss
            elbo_loss.backward()
            train_loss += elbo_loss
            epoch_train_losses.append(elbo_loss.item())
            optimizer.step()
            scheduler.step()

            batch_train_tqdm.set_postfix(loss=elbo_loss.item())

        model.eval()

        with torch.no_grad():
            for batch_data in batch_test_tqdm :

                # Looping over and calculating stuff
                data, labels = batch_data
                data = data.to(model.device)
                labels = labels.to(model.device)
                z_batch, x_mu_batch, posterior_params = model(data)

                # step
                elbo_loss = model._elbo(data, x_mu_batch, posterior_params)
                test_loss += elbo_loss
                epoch_test_losses.append(elbo_loss.item())

                batch_test_tqdm.set_postfix(loss=elbo_loss.item())

        avg_train = np.mean(epoch_train_losses)
        avg_test = np.mean(epoch_test_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Finished epoch {epoch+1}/{n_epochs} | Loss: {avg_train:.4f}")
            torch.save(model.state_dict(), f'./saves/pusht_chkpt_{epoch + 1}.pth')



    

    