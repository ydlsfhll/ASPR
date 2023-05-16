from torch import optim
from LARC import LARC
# from timm.optim.lars import Lars
from adan import Adan


def get_optimizer(args, params):
    parameters = {'params':params, 'lr':args.learning_rate}
    optimizer = None
    
    if args.optimizer == 'SGD':
        parameters['momentum'] = 0.9
        if args.nesterov:
            parameters['nesterov'] = True
        parameters['weight_decay'] = 5e-4
        optimizer = optim.SGD(**parameters)

    elif args.optimizer == 'Adagrad':
        parameters['weight_decay'] = 2e-5
        optimizer = optim.Adagrad(**parameters)
        
    elif args.optimizer == 'AdamW':
        parameters['weight_decay'] = 1e-4
        optimizer = optim.AdamW(**parameters)

    elif args.optimizer == 'Lars':
        parameters['momentum'] = 0.905
        parameters['weight_decay'] = 2e-5
        parameters['nesterov'] = True
        parameters['trust_clip'] = True
        optimizer = Lars(**parameters)
        
    if args.lars:
        optimizer = LARC(optimizer, 0.001, False)
    
    return optimizer
