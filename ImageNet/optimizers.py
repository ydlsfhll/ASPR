from torch import optim
from adan import Adan


def get_optimizer(args, params):
    parameters = {'params':params, 'lr':args.learning_rate}
    optimizer = None
    
    if args.optimizer == 'SGD':
        parameters['momentum'] = 0.9
        parameters['weight_decay'] = 5e-4
        # parameters['nesterov'] = True
        optimizer = optim.SGD(**parameters)

    elif args.optimizer == 'Adagrad':
        parameters['weight_decay'] = 2e-5
        optimizer = optim.Adagrad(**parameters)
        
    elif args.optimizer == 'AdamW':
        parameters['weight_decay'] = 1e-4
        optimizer = optim.AdamW(**parameters)
        
    elif args.optimizer == 'Adan':
        optimizer = Adan(**parameters)
    
    return optimizer
