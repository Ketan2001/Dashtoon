import numpy as np
import torch
from utils import *
from transformer import TransformerNet
from vgg import PerceptualLossNet
from torch.optim import Adam
import time
from PIL import Image
import os
import pickle
from torch.utils.data import random_split

# GLOBAL SETTINGS
NUM_TRAIN_IMAGES = 50000  # Set to -1 to use all (~83k) Images
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "./data"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "./style_image.jpeg"
BATCH_SIZE = 4
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 4e5
TV_WEIGHT = 1e-6
LR = 0.001
SAVE_MODEL_PATH = "./checkpoints"
SAVE_IMAGE_PATH = "./image_outputs"
SAVE_MODEL_EVERY = 500  # 2,000 Images with batch size 4
SEED = 42
CHECKPOINT_FREQ = 500
LOG_FREQ = 250


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data loader
    print("Loading Data...")
    train_loader, val_loader = get_training_data_loader(DATASET_PATH, TRAIN_IMAGE_SIZE, BATCH_SIZE)
    print("Data Loaded Successfully \n")

    # prepare neural networks
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    optimizer = Adam(transformer_net.parameters())

    # Calculate style image's Gram matrices (style representation)
    # Built over feature maps as produced by the perceptual net - VGG16
    style_img_path = STYLE_IMAGE_PATH
    style_img = prepare_img(STYLE_IMAGE_PATH, device=device, batch_size=BATCH_SIZE)
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [gram_matrix(x) for x in style_img_set_of_feature_maps]

    test_image = prepare_img('content.jpeg', device)

    lowest_loss = None

    history = {
        'content_loss': [],
        'style_loss': [],
        'tv_loss': [],
        'total_loss': []
    }

    print("Started Training...")
    ts = time.time()
    for epoch in range(NUM_EPOCHS):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # step1: Feed content batch through transformer net
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # step2: Feed content and stylized batch through perceptual net (VGG16)
            content_batch_set_of_feature_maps = perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = perceptual_loss_net(stylized_batch)

            # step3: Calculate content representations and content loss
            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
            mse_loss = torch.nn.MSELoss(reduction='mean')
            content_loss = CONTENT_WEIGHT * mse_loss(target_content_representation, current_content_representation)

            # step4: Calculate style representation and style loss
            style_loss = 0.0
            current_style_representation = [gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += mse_loss(gram_gt, gram_hat)
            style_loss /= len(target_style_representation)
            style_loss *= STYLE_WEIGHT

            # step5: Calculate total variation loss - enforces image smoothness
            tv_loss = TV_WEIGHT * total_variation(stylized_batch)

            # step6: Combine losses and do a backprop
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            optimizer.zero_grad()  # clear gradients for the next round

            with torch.no_grad():
                history['content_loss'].append(content_loss.item())
                history['style_loss'].append(style_loss.item())
                history['tv_loss'].append(tv_loss.item())
                history['total_loss'].append(total_loss.item())

                if (batch_id + 1) % LOG_FREQ == 0:
                    print(
                        f'Iter : [{batch_id + 1}/{len(train_loader)}] \n ------------- \n time elapsed={(time.time() - ts) / 60:.2f} \n c-loss={acc_content_loss / LOG_FREQ}|s-loss={acc_style_loss / LOG_FREQ}|tv-loss={acc_tv_loss / LOG_FREQ}|total loss={(acc_content_loss + acc_style_loss + acc_tv_loss) / LOG_FREQ}')
                    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

                    transformer_net.eval()
                    stylized_test = transformer_net(test_image).cpu().numpy()[0]
                    transformer_net.train()
                    stylized = post_process_image(stylized_test)
                    stylized_image = Image.fromarray(stylized)

                    stylized_image.save(os.path.join(SAVE_IMAGE_PATH, f"iter-{batch_id + 1}.jpeg"))

                if (batch_id + 1) % CHECKPOINT_FREQ == 0:
                    torch.save(transformer_net.state_dict(),
                               os.path.join(SAVE_MODEL_PATH, f"Iter-{batch_id + 1}-{total_loss.item() : .4f}.pth"))

                if lowest_loss is None or total_loss.item() < lowest_loss:
                    lowest_loss = total_loss.item()
                    torch.save(transformer_net.state_dict(),
                               f"best_model.pth")


if __name__ == "__main__":
    train()
