"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'use_full': bool,
        'start_from_pretrained_model': bool,
        'full':
            {'paths': {
                'pretrained_model_path': str,
                'base_path': str,
                'logs': str,
                'vocab_path': str, },
            'train_paths': {
                'vqaDataset': str,
                'questions': str,
                'answers': str,
                'imgs': str,
                'processed_imgs': str},
            'val_paths': {
                'vqaDataset': str,
                'questions': str,
                'answers': str,
                'imgs': str,
                'processed_imgs': str}},
        'small':
            {'paths': {
                'pretrained_model_path': str,
                'base_path': str,
                'logs': str,
                'vocab_path': str, },
            'train_paths': {
                'vqaDataset': str,
                'questions': str,
                'answers': str,
                'imgs': str,
                'processed_imgs': str},
            'val_paths': {
                'vqaDataset': str,
                'questions': str,
                'answers': str,
                'imgs': str,
                'processed_imgs': str}}
    },
    'train': {
        'text': {
            'question_features': int,
            'embedding_features': int,
            'dropout': float,
            'num_lstm_layers': int,
            'bidirectional': bool,
        },
        'image': {
            # 'image_features': int,
            'kernel_size': int,
            'dropout': float,
            'num_channels': list,
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

        'n_epochs_stop': int,
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
