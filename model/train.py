from spherical_vae import *
from dataloader.utils import *
from torch.optim.adam import adam 
from tqdm import tqdm
# -------------------------- Training Loop ----------------------------

def train(model, train_loader, test_loader, optimizer, scheduler):
    """
    Training the model
    """
    # losses

    batch_tqdm = tqdm(train_loader)
    
    n_epochs = model.config["n_epochs"]
    
    batch_train_tqdm = tqdm(train_loader, desc='Batch Train')
    batch_test_tqdm = tqdm(test_loader, desc='Batch Loss')
    tglobal = tqdm(range(n_epochs), desc='Epoch', leave=False)

    for epoch in tglobal:
            
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
            epoch_train_losses.append(train_loss)
            optimizer.step()
            scheduler.step()

            elbo_loss_cpu = elbo_loss.item()
            batch_train_tqdm.set_postfix(elbo_loss_cpu)

        tglobal.set_postfix(loss = np.mean(epoch_train_losses))

        print(f"Beginning Eval mode !")
        model.eval()

        with torch.no_grad():
            for batch_data in batch_test_tqdm :

                # Looping over and calculating stuff
                data, labels = batch_data
                data = data.to(model.device)
                labels = labels.to(model.device)
                z_batch, x_mu_batch, posterior_params = model(data)

                elbo_loss = model._elbo(data, x_mu_batch, posterior_params)
                tets_loss += elbo_loss
                epoch_test_losses.append(test_loss)
                elbo_loss_cpu = elbo_loss.item()
                batch_test_tqdm.set_postfix(elbo_loss_cpu)

        tglobal.set_postfix(loss = np.mean(epoch_test_losses))
        
        if (epoch + 1) % 10 == 0:
                print(f"Finished epoch {epoch+1}/{n_epochs} | Loss: {np.mean(epoch_train_losses):.4f}")
                torch.save(model.state_dict(), f'./saves/pusht_chkpt_{epoch + 1}.pth')


    

    