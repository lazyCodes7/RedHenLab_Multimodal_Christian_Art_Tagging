from trainer import Trainer
import argparse
import torch
def train(device, batch_size, data_dir, metadata_path, feature_extractor_path, epochs, lr):
    trainer = Trainer(
        device,
        batch_size, 
        data_dir,
        metadata_path,
        feature_extractor_path,
        epochs, 
        lr
    )

    # Training function
    trainer.train()

def test(device, batch_size, data_dir, metadata_path, feature_extractor_path, epochs, lr, model_path):
    trainer = Trainer(
        device,
        batch_size, 
        data_dir,
        metadata_path,
        feature_extractor_path,
        epochs, 
        lr
    )
    trainer.model.load_state_dict(torch.load(model_path, map_location = device))

    # Testing function
    trainer.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=False, default = '../modules/captioning/metadata_v2.csv')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--feature_extractor_path', type=str, required=True)
    parser.add_argument('-e', '--epochs', type = int, default = 2)
    parser.add_argument('-lr', '--learning_rate', type = int, default = 1e-4)
    parser.add_argument('-d', '--device', type = str, default = 'cuda')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_batch_size', type = int, default = 16)
    parser.add_argument('--captioning_model_path', type = str, default = '')
    
    args = parser.parse_args()

    if(args.train):
        train(
            args.device,
            args.train_batch_size,
            args.data_dir,
            args.metadata_path,
            args.feature_extractor_path,
            args.epochs, 
            args.learning_rate
        )
    else:
        test(
            args.device,
            args.train_batch_size,
            args.data_dir,
            args.metadata_path,
            args.feature_extractor_path,
            args.epochs, 
            args.learning_rate,
            args.captioning_model_path
        )



