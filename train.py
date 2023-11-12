import numpy as np
import torch
from utils import *

# Outlined in Johnson et al.
from transformer import TransformerNet

# pretrained VGG to extract features form relu_1_2, relu_2_2, relu_3_3 and relu_4_3
from vgg import PerceptualLossNet

from torch.optim import Adam
import time
from PIL import Image
import os
import pickle
from torch.utils.data import random_split

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Hyperparameters
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "./data"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "./style_image.jpeg"
CONTENT_IMAGE_PATH = "content.jpeg"
BATCH_SIZE = 4
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 4e5
TV_WEIGHT = 1e-6
LR = 0.001
SAVE_MODEL_PATH = "./checkpoints"
SAVE_IMAGE_PATH = "./image_outputs"
CHECKPOINT_FREQ = 150
LOG_FREQ = 50

# Setting the seed value for reproducibility
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class StyleTransfer:
    def __init__(
        self,
        num_epochs=NUM_EPOCHS,
        image_size=TRAIN_IMAGE_SIZE,
        dataset_path=DATASET_PATH,
        style_image_path=STYLE_IMAGE_PATH,
        content_image_path=CONTENT_IMAGE_PATH,
        batch_size=BATCH_SIZE,
        style_weight=STYLE_WEIGHT,
        content_weight=CONTENT_WEIGHT,
        tv_weight=TV_WEIGHT,
        log_freq=LOG_FREQ,
        checkpoint_freq=CHECKPOINT_FREQ,
        lr=LR,
        save_model_path=SAVE_MODEL_PATH,
        save_image_path=SAVE_IMAGE_PATH
    ):
        self.epochs = num_epochs
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.style_image_path = style_image_path
        self.content_image_path = content_image_path
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.lr = lr
        self.log_freq = log_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_model_path = save_model_path
        self.save_image_path = save_image_path

        # load data
        print("Loading Data...")
        self.train_loader, self.val_loader = get_training_data_loader(dataset_path, image_size, batch_size)
        print("Data Loaded Successfully \n")
        
        # instantiate networks
        self.transformer_net = TransformerNet().train().to(self.device)
        self.perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(self.device)

        self.optimizer = Adam(self.transformer_net.parameters())

        # Compute Gram matrices for the style image
        style_img = prepare_img(style_image_path, self.device, batch_size=batch_size)
        style_img_set_of_feature_maps = self.perceptual_loss_net(style_img)
        self.target_style_representation = [gram_matrix(x) for x in style_img_set_of_feature_maps]
        
        # This image is used to keep track of the subjective performance over the iteration
        # sotred in the directory "save_image_path" after every log_iter iterations
        self.test_image = prepare_img(content_image_path, self.device)
        
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        self.best_val_loss = None
        
        self.history = {
            'content_loss_t': [],
            'style_loss_t': [],
            'tv_loss_t': [],
            'total_loss_t': [],
            'total_loss_v' : []
        }
    
    
    def train(self):
        print("Training Started...\n")
        
        ts = time.time()
        
        acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
        
        for epoch in range(self.epochs):
            for batch_id, (content_batch, _) in enumerate(self.train_loader):
                # get ouput of transform_net
                content_batch = content_batch.to(self.device)
                stylized_batch = self.transformer_net(content_batch)

                # feed batch of style and content images to the vgg net
                content_batch_set_of_feature_maps = self.perceptual_loss_net(content_batch)
                stylized_batch_set_of_feature_maps = self.perceptual_loss_net(stylized_batch)

                # compute content loss
                target_content_representation = content_batch_set_of_feature_maps.relu2_2
                current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
                content_loss = self.content_weight * self.mse_loss(target_content_representation, current_content_representation)
                acc_content_loss += content_loss.item()
                
                # compute gram matrices and style loss
                style_loss = 0.0
                current_style_representation = [gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
                for gram_gt, gram_hat in zip(self.target_style_representation, current_style_representation):
                    style_loss += self.mse_loss(gram_gt, gram_hat)
                style_loss /= len(self.target_style_representation)
                style_loss *= self.style_weight
                acc_style_loss += style_loss.item()

                # compute tv loss
                tv_loss = self.tv_weight * total_variation(stylized_batch)
                acc_tv_loss += tv_loss.item()

                # backprop
                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()
                
                if (batch_id + 1) % self.log_freq == 0:
                    with torch.no_grad():
                        self.history['content_loss_t'].append(acc_content_loss / self.log_freq)
                        self.history['style_loss_t'].append(acc_style_loss / self.log_freq)
                        self.history['tv_loss_t'].append(acc_tv_loss / self.log_freq)
                        self.history['total_loss_t'].append((acc_content_loss + acc_style_loss + acc_tv_loss) / self.log_freq)

                        self.transformer_net.eval()
                        stylized_test = self.transformer_net(self.test_image).cpu().numpy()[0]
                        val_loss = self.val_loss()
                        self.transformer_net.train()
                        stylized = post_process_image(stylized_test)
                        stylized_image = Image.fromarray(stylized)

                        stylized_image.save(os.path.join(self.save_image_path, f"iter-{batch_id + 1}.jpeg"))
                        
                        
                        
                        if self.best_val_loss is None or val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            torch.save(self.transformer_net.state_dict(), "best_model.pth")
                            
                        print(f'Iter : [{batch_id + 1}/{len(self.train_loader)}]')
                        print('---------------------\n')
                        print(f'Time Elapsed : {(time.time() - ts) / 60:.2f} min)')
                        print('Training Loss :')
                        print(f'\tContent Loss : {acc_content_loss / self.log_freq}')
                        print(f'\tStyle Loss : {acc_style_loss / self.log_freq}')
                        print(f'\tTV Loss : {acc_tv_loss / self.log_freq}')
                        print(f'\tTotal Loss : {(acc_content_loss + acc_style_loss + acc_tv_loss) / self.log_freq}')
                        print(f'Validation Loss : {val_loss}\n\n')
                    
                        
                        acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

                
                if (batch_id + 1) % self.checkpoint_freq == 0:
                    torch.save(self.transformer_net.state_dict(),
                               os.path.join(self.save_model_path, f"iter-{batch_id + 1}.pth"))


                            
    def val_loss(self):
        val_loss = 0.0
        for batch_id, (content_batch, _) in enumerate(self.val_loader):
            content_batch = content_batch.to(self.device)
            stylized_batch = self.transformer_net(content_batch)
            
            content_batch_set_of_feature_maps = self.perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = self.perceptual_loss_net(stylized_batch)
            
            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
            content_loss = self.content_weight * self.mse_loss(target_content_representation, current_content_representation)

            style_loss = 0.0
            current_style_representation = [gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(self.target_style_representation, current_style_representation):
                style_loss += self.mse_loss(gram_gt, gram_hat)
            style_loss /= len(self.target_style_representation)
            style_loss *= self.style_weight
            
            tv_loss = self.tv_weight * total_variation(stylized_batch)

            val_loss += (content_loss + style_loss + tv_loss).item()
            
        val_loss /= len(self.val_loader)
        self.history['total_loss_v'].append(val_loss)
                
        return val_loss

if __name__ == "__main__":
    style_transfer = StyleTransfer()
    style_transfer.train()
