class Config:
    # Define loss names
    LOSS_NAMES = {
        "m_loss", "i_loss", "u_loss", "v_loss"
    }

    LOSS_WEIGHTS = {
        "m_loss": 1.,
        "i_loss": 1.,
        "u_loss": 4.,
        "v_loss": 4.
    }

    LEARNING_RATE = 0.002

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001