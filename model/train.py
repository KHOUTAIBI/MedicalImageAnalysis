from model.spherical_vae import *
from dataloader.utils import * 
from tqdm import tqdm

# -------------------------- Training Loop ----------------------------

def train(model, train_loader, test_loader, optimizer, scheduler):
    """
    Training the model
    """
    
    n_epochs = model.config["n_epochs"]
    tglobal = tqdm(range(n_epochs), desc='Epoch')
    
    
    for epoch in tglobal:

        avg_train, avg_test = train_epoch(model, train_loader, test_loader, optimizer, scheduler, epoch)
        tglobal.set_postfix(loss_train = avg_train, loss_test = avg_test)



def train_epoch(model, train_loader, test_loader, optimizer, scheduler, epoch):
    
    batch_train_tqdm = tqdm(train_loader, desc='Batch Train', leave=False)
    batch_test_tqdm = tqdm(test_loader, desc='Batch Test', leave=False)

    epoch_train_losses = []
    epoch_test_losses = []

    # -------------------------------------- TRAIN ----------------------------------------------
    model.train()
    
    for data, labels in batch_train_tqdm:

        data = data.to(model.device)
        labels = labels.to(model.device)

        optimizer.zero_grad()

        if model.config["dataset"] == "S2_dataset" or model.config["dataset"] == "S1_dataset":

            z_batch, x_mu_batch, posterior_params = model(data)
            z_proj = z_batch / z_batch.norm(dim = -1, keepdim = True)
            z_mu, _ = posterior_params

            elbo_loss_train = model._elbo(data, x_mu_batch, posterior_params)
            latent_loss_batch_train = latent_loss(labels, z_proj, model.config)
            elbo_loss_train += model.config["alpha"] * latent_loss_batch_train

            if epoch % 50 == 0:
                z_mu, z_kappa = posterior_params
                q_z = VonMisesFisher3D(z_mu, z_kappa)
                kappa = q_z.scale

                recon = torch.mean((data - x_mu_batch) ** 2).item()
                KL = model.kl_vmf_spherical_uniform(kappa).mean().item()
                latent_loss_mean = latent_loss(labels, z_batch, model.config).mean()

                print(
                    f"[DEBUG] recon={model.config['gamma'] * recon:.4f}  KL={KL:.4f}  "
                    f"beta*KL={model.config['beta'] * KL:.4f}  "
                    f"kappa_mean={z_kappa.mean().item():.4f} "
                    f"latent_loss_mean={model.config['alpha'] * latent_loss_mean}"
                )

        elif model.config["dataset"] == "T2_dataset":

            z_batch, x_mu_batch, posterior_params = model(data)
            z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params

            z_latent = model._build_torus(z_theta_mu, z_phi_mu)

            elbo_loss_train = (
                model._elbo(data, x_mu_batch, (z_theta_mu, z_theta_kappa)) +
                model._elbo(data, x_mu_batch, (z_phi_mu, z_phi_kappa))
            )

            latent_loss_batch_train = latent_loss(labels, z_latent, model.config)
            elbo_loss_train += model.config["alpha"] * latent_loss_batch_train

            if epoch % 50 == 0:

                q_theta_z = VonMisesFisher3D(z_theta_mu, z_theta_kappa)
                q_phi_z = VonMisesFisher3D(z_phi_mu, z_phi_kappa)
                kappa_theta = q_theta_z.scale
                kappa_phi = q_phi_z.scale

                recon = torch.mean((data - x_mu_batch) ** 2).item()
                KL = (
                    model.kl_vmf_spherical_uniform(kappa_theta).mean().item() +
                    model.kl_vmf_spherical_uniform(kappa_phi).mean().item()
                )
                latent_loss_mean = latent_loss(labels, z_latent, model.config).mean()

                print(
                    f"[DEBUG] recon={model.config['gamma'] * recon:.4f}  KL={KL:.4f}  "
                    f"beta*KL={model.config['beta'] * KL:.4f}  "
                    f"theta_kappa_mean={z_theta_kappa.mean().item():.4f} "
                    f"latent_loss_mean={model.config['alpha'] * latent_loss_mean}"
                )

        elbo_loss_train.backward()
        optimizer.step()

        epoch_train_losses.append(elbo_loss_train.item())
        batch_train_tqdm.set_postfix(loss_train=elbo_loss_train.item())

    # ------------------------------------------ TEST ----------------------------------------
    model.eval()
    with torch.no_grad():
        for data, labels in batch_test_tqdm:
            data = data.to(model.device)
            labels = labels.to(model.device)

            z_batch, x_mu_batch, posterior_params = model(data)

            if model.config["dataset"] == "T2_dataset":
                z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
                z_latent = model._build_torus(z_theta_mu, z_phi_mu)
                elbo_loss_test = (
                    model._elbo(data, x_mu_batch, (z_theta_mu, z_theta_kappa)) +
                    model._elbo(data, x_mu_batch, (z_phi_mu, z_phi_kappa))
                )
                latent_loss_batch_test = latent_loss(labels, z_latent, model.config)
            else:
                elbo_loss_test = model._elbo(data, x_mu_batch, posterior_params)
                latent_loss_batch_test = latent_loss(labels, z_batch, model.config)

            elbo_loss_test += model.config["alpha"] * latent_loss_batch_test
            epoch_test_losses.append(elbo_loss_test.item())

            batch_test_tqdm.set_postfix(loss_test=elbo_loss_test.item())

    avg_train = np.mean(epoch_train_losses)
    avg_test = np.mean(epoch_test_losses)

    if model.config["scheduler"]:
        scheduler.step(avg_train)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), model.config["save_path"])

    return avg_train, avg_test


    