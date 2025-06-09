**Regularization Cocktails**:
- [x] `batch_normalization`
    - Description: If batch normalization regularization should be used.
    - Value: `[True, False]`
- [x] `stochastic_weight_averaging`
    - Description: If stochastic weight averaging should be used.
    - Observation: Actually, this has some hyperparameters itself (start_epoch, lr_schedule,  swa_lr, swa_lr2, swa_freq)
- [x] `lookahead`
    - Description: If the lookahead optimizing technique should be used.
    - Value: `[True, False]`
    - Hyperparameters:
        - `learning_rate`
        - `num_steps`

- [x] `regularization_factor` (weight decay)
    - Description: If weight decay regularization should be used.
    - Value: `[0, 1]`
    - Remarks: Currently on a linear scale, could maybe be changed to log scale.
    
- [x] `dropout` 
    - Description: If dropout regularization should be used.
    - Value: `[0, 1]`
    - Remarks: No special dropout shapes are considered as in https://proceedings.neurips.cc/paper_files/paper/2021/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf
    - 
- [x] `snapshot_ensembling`
    - Description: If snapshot ensembling should be used.
    - Value: `[True, False]`
    - Hyperparameters:
        - `period_init`
        - `period_increase`
        - `reset_weights`

- [x] `skip_connection`
    - Description: If skip connections should be used. Turns the network into a residual network.
    - Value: `[True, False]`

- [X] `shake_shake`
    - Description: Trains two independent blocks instead of a single one but perturbs them through random fusion and randomly mixed gradients.
    - Value: `[True, False]`

- [X] `shake_drop`
    - Description: Alters a block with two random operation: One checks whether the block is discarded completely (only skip signal is used) and one randomly alters the outcome with a random factor. Due to the probability of signal loss, this technique requires skip connections.
    - Value: `[True, False]`

- [x] `augmentation`
    - Description: If methods that augment examples should be used.
    - Remarks: Can be implemented at dataset or batch level
    - Methods:
        - `cutout`: Implemented a dataset level (with a probability for a datum being set to 0)
        - `mixup`: Implemented a dataset level
        - `cutmix`: Implemented a dataset level
        - `adversarial`: Not implemented
