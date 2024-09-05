from matplotlib import pyplot as plt


def draw_loss_history(losses, model_name):
    ae1_losses, ae2_losses = zip(*losses)
    sum_losses = [x + y for x, y in zip(ae1_losses, ae2_losses)]
    plt.plot(ae1_losses, label='AE1')
    plt.plot(ae2_losses, label='AE2')
    plt.plot(sum_losses, label='Sum')
    plt.legend()
    plt.savefig(model_name + '_loss.png')
    plt.clf()


def draw_loss_history_for_single_loss(losses, model_name):
    plt.plot(losses, label='loss')
    plt.legend()
    plt.savefig(model_name + '_loss.png')
    plt.clf()


def draw_loss_history_for_mad_gan(G_losses, D_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_name + '_loss.png')
    plt.clf()
