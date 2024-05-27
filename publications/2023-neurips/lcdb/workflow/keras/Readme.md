**Regularization Cocktails**:
- [x] `batch_normalization`
    - Description: If batch normalization regularization should be used.
    - Value: `[True, False]`
- [x] `stochastic_weight_averaging`
    - Description: If stochastic weight averaging should be used. This is implemented via the `keras-swa` package: https://github.com/simon-larsson/keras-swa
    - Value: `[True, False]`
    - Observation: Actually, this has some hyperparameters itself (start_epoch, lr_schedule,  swa_lr, swa_lr2, swa_freq)
- [x] `lookahead`
    - Description: If the lookahead optimizing technique should be used.
    - Value: `[True, False]`

- [x] `weight_decay`
    - Description: If weight decay regularization should be used.
    - Value: `[True, False]`
    
- [x] `dropout` 
    - Description: If dropout regularization should be used.
    - Value: `[True, False]`
- [x] `snapshot_ensembling`
    - Description: If snapshot ensembling should be used.
    - Value: `[True, False]`

- [x] `skip_connection`
    - Description: If skip connections should be used. Turns the network into a residual network.
    - Value: `[True, False]`

- [ ] `multi_branch_choice`
    - Description: Multibranch network regularization. Only active when `skip_connection` is active.
    - Value: `['none', 'shake-shake', 'shake-drop']`

- [ ] `augmentation`
    - Description: If methods that augment examples should be used.
    - Value: `['mixup', 'cutout', 'cutmix', 'standard', 'adversarial']`
