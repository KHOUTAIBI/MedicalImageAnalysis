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
    tglobal = tqdm(range(n_epochs), desc='Epoch')
    
    for epoch in tglobal:

        batch_train_tqdm = tqdm(train_loader, desc='Batch Train', leave=False)
        batch_test_tqdm = tqdm(test_loader, desc='Batch Test', leave=False)

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
            

            elbo_loss_train = model._elbo(data, x_mu_batch, posterior_params)
            latent_loss_batch_train = latent_loss(labels, z_batch, model.config)
            elbo_loss_train += model.config["alpha"] * latent_loss_batch_train


            # ! PROBLEM : KAPPA INCREASES / THE POLICY BECOMES DETERMINISTIC AS kappa->inifnity / THIS IS BECAUSE OF THE DATASET DETERMINISTIC !!!
            if epoch % 50 == 0 :  # first batch of first epoch
                z_mu, z_kappa = posterior_params
                q_z = VonMisesFisher3D(z_mu, z_kappa)
                kappa = q_z.scale

                recon = torch.mean((data - x_mu_batch) * (data - x_mu_batch)).item()
                KL = model.kl_vmf_spherical_uniform(kappa).mean().item()

                print(
                    f"[DEBUG] recon={recon:.4f}  KL={KL:.4f}  "
                    f"beta*KL={model.config['beta']*KL:.4f}  "
                    f"kappa_mean={z_kappa.mean().item():.4f}"
                )

            # loss
            elbo_loss_train.backward()
            train_loss += elbo_loss_train
            epoch_train_losses.append(elbo_loss_train.item())
            optimizer.step()
            

            batch_train_tqdm.set_postfix(loss_train=elbo_loss_train.item())

        # step scheduler
        if model.config["scheduler"]:
                scheduler.step()
        model.eval()
        
        with torch.no_grad():
            for batch_data in batch_test_tqdm :

                # Looping over and calculating stuff
                data, labels = batch_data
                data = data.to(model.device)
                labels = labels.to(model.device)
                z_batch, x_mu_batch, posterior_params = model(data)

                # step
                elbo_loss_test = model._elbo(data, x_mu_batch, posterior_params)
                latent_loss_batch_test = latent_loss(labels, z_batch, model.config)
                elbo_loss_test += model.config["alpha"] * latent_loss_batch_test
                test_loss += elbo_loss_test
                epoch_test_losses.append(elbo_loss_test.item())

                batch_test_tqdm.set_postfix(loss_test=elbo_loss_test.item())

        avg_train = np.mean(epoch_train_losses)
        avg_test = np.mean(epoch_test_losses)
        tglobal.set_postfix(loss_train = avg_train, loss_test = avg_test)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./saves/spherical_VAE_chkpt_final.pth')



    

    