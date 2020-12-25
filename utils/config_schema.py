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
            'processed_imgs': str,
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
        'question_features': int,
        'image_features': int,
        'classifier_hidden_dim': int,
        'attention_hidden_dim': int,
        'glimpses': int,
        'max_answers': int,
        'embedding_features': int,
        'num_epochs': int,
        'image_size': int,
        'central_fraction': float,
        'dropouts': {
            'text': float,
            'attention': float,
            'classifier': float,
        },
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
