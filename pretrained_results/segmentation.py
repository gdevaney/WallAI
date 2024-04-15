import torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
import os

from semantic_segmentation_pytorch.mit_semseg.models import ModelBuilder, SegmentationModule
from semantic_segmentation_pytorch.mit_semseg.utils import colorEncode

def return_semantic_result(filename: str):
    print('hihih: ', os.getcwd())
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='/Users/yoon/Documents/gatech/spring24/WallAI/semantic_segmentation_pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='/Users/yoon/Documents/gatech/spring24/WallAI/semantic_segmentation_pytorch/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_module.to(DEVICE)
    # segmentation_module.cuda()

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    # pil_image = PIL.Image.open('our_data/Test/33012f501af95eecd41803710646e57e87c2680d.jpg').convert('RGB')
    pil_image = PIL.Image.open(filename).convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    # singleton_batch = {'img_data': img_data[None].cuda()}
    singleton_batch = {'img_data': img_data[None].to(DEVICE)}
    output_size = img_data.shape[1:]


    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    def visualize_wall(img, pred, class_to_display=0):
        """
            Function for visualizing wall prediction
            (original image, segmentation mask and original image with the segmented wall)
        """
        img_green = img.copy()
        black_green = img.copy()
        img_green[pred == class_to_display] = [111, 209, 201]
        black_green[pred == class_to_display] = [111, 209, 201]
        black_green[pred != class_to_display] = [0, 0, 0]
        # im_vis = numpy.concatenate((img, black_green, img_green), axis=1)
        # im_vis = numpy.concatenate((img, img_green), axis=1)
        return PIL.Image.fromarray(img_green)

        # display(PIL.Image.fromarray(im_vis))

    predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
    # return visualize_wall(img_original, pred, predicted_classes[0])
    return visualize_wall(img_original, pred)
    
    # for c in predicted_classes[:15]:
    #     visualize_wall(img_original, pred, c)