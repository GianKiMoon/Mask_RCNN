class Config:
    # Define loss names
    LOSS_NAMES = {
        "m_loss", "i_loss", "u_loss", "v_loss"
    }

    LOSS_WEIGHTS = {
        "m_loss": 0.2,
        "i_loss": 0.5,
        "u_loss": 8.,
        "v_loss": 8.
    }

    LEARNING_RATE = 0.002

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    MOMENTUM = 0.9
