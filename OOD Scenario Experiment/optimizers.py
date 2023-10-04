from torch import optim


def get_optimizer(args, params):
    parameters = {'params':params, 'lr':args.learning_rate}
    optimizer = None
    
    if args.optimizer == 'SGD':
        parameters['momentum'] = 0.9
        parameters['nesterov'] = True
        parameters['weight_decay'] = 5e-4
        optimizer = optim.SGD(**parameters)
        
    elif args.optimizer == 'AdamW':
        parameters['weight_decay'] = 0.
        optimizer = optim.AdamW(**parameters)
    
    return optimizer
