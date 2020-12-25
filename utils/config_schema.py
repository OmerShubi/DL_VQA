"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'paths': {
            'base_path': str,
            'logs': str,
            'vocab_path': str, },
        'train_paths': {
            'questions': str,
            'answers': str,
            'imgs': str,
            'processed_imgs': str},
        'val_paths': {
            'questions': str,
            'answers': str,
            'imgs': str,
            'processed_imgs': str}
    },
    'train': {
        'text': {
            'question_features': int,
            'embedding_features': int,
            'dropout': float,
        },
        'image': {
            'image_features': int,
        },
        'attention': {
            'hidden_dim': int,
            'glimpses': int,
            'dropout': float,

        },
        'classifier': {
            'hidden_dim': int,
            'dropout': float,
        },

        'max_answers': int,
        'image_size': int,
        'central_fraction': float,

        'num_epochs': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
