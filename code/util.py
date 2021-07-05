import models

def get_model(modname, **kwargs):
    input_dim = kwargs['input_dim']
    big_dim = kwargs['big_dim']
    hidden_dim = kwargs['hidden_dim']
    emd_modname = kwargs['emd_modname']

    if modname == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        if modname[-3:] == 'EMD':
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=emd_modname)
        else:
            model = getattr(models, modname)(input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim)
    return model
