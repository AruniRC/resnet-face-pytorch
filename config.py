


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    ),

    # python train_*.py -g 0 -c 1
    1: dict(
        max_iteration=500,
        lr=1.0e-3, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=10,
        optim='Adam',
    ),

    2: dict(
        max_iteration=100,
        lr=1.0e-3, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=10,
        optim='Adam',
    ),

}



