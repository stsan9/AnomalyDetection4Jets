import models

def get_model(modname):
    if modname == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        if modname[-3:] == 'EMD':
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)
        else:
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    return model
