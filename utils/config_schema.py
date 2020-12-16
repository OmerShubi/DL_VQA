"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'paths': {
            'base_path': str,
            'logs': str,
            'processed_imgs': str,
            'vocab_path': str, },
        'train_paths': {
            'questions': str,
            'answers': str,
            'imgs': str},
        'val_paths': {
            'questions': str,
            'answers': str,
            'imgs': str}
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
        'grad_clip': float,
        'dropouts': {
            'text': float,
            'attention': float,
            'classifier': float,
        },
        'num_hid': int,
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
