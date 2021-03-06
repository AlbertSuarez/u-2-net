import argparse
import glob
import os
import time
import warnings
import torch

from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from src.u2_net.data_loader import RescaleT
from src.u2_net.data_loader import ToTensorLab
from src.u2_net.data_loader import SalObjDataSet
from src.u2_net.model_enum import Model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--model', type=str, default=Model.u2net, choices=Model.list())
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--show_user_warnings', action='store_true')
    return parser.parse_args()


def _load_data(input_path_list):
    """
    Create DataLoader instance form input path list.
    :param input_path_list: Input path list.
    :return: DataLoader instance.
    """
    data_set = SalObjDataSet(
        img_name_list=input_path_list,
        lbl_name_list=[],
        trans=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    return data_loader


def _define_model(model, model_path, gpu):
    """
    Define model given some parameters.
    :param model: Model enumeration.
    :param model_path: Model file path.
    :param gpu: If GPU is available or not.
    :return: Model instance.
    """
    net = model.value()
    if gpu:
        net.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net


# noinspection PyUnresolvedReferences
def _normalize_prediction(prediction):
    """
    Normalize the predicted SOD probability map.
    :param prediction: Model prediction.
    :return: Prediction normalized.
    """
    maximum = torch.max(prediction)
    minimum = torch.min(prediction)
    prediction_normalized = (prediction - minimum) / (maximum - minimum)
    return prediction_normalized


def _save_output(input_image_name, prediction, output_folder_path):
    """
    Save output given the prediction.
    :param input_image_name: Input image name.
    :param prediction: Prediction from model.
    :param output_folder_path: Output folder path where the image will be saved.
    :return: Output image saved.
    """
    # Prepare prediction
    prediction = prediction.squeeze()
    prediction = prediction.cpu().data.numpy()
    # Generate output image
    image = io.imread(input_image_name)
    prediction_image = Image.fromarray(prediction * 255).convert('RGB')
    output_image = prediction_image.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    # Save output
    file_name_extension_split = input_image_name.split('/')[-1].split('.')
    file_name = '_'.join(file_name_extension_split[0:-1])
    os.makedirs(output_folder_path, exist_ok=True)
    output_image.save(os.path.join(output_folder_path, f'{file_name}.png'))


def run(input_path, output_path, model, gpu, show_user_warnings):
    """
    Run inference using U^2-Net model.
    :param input_path: Input path (image or folder).
    :param output_path: Output path (has to be a folder).
    :param model: Model to use.
    :param gpu: If GPU is available or not.
    :param show_user_warnings: If UserWarnings have to be showed or not.
    :return: All images processed.
    """
    if not show_user_warnings:
        warnings.simplefilter('ignore', UserWarning)
    if os.path.exists(output_path) and os.path.isfile(output_path):
        print(f'Output path exists and it is a file: [{output_path}].')
        return
    if os.path.exists(input_path):
        # Get input path list
        if os.path.isfile(input_path):
            input_path_list = [input_path]
        elif os.path.isdir(input_path):
            input_path_list = glob.glob(os.path.join(input_path, '*'))
        else:
            print(f'Input path specified is not a file or a folder: [{input_path}].')
            return
        print(f'Files to process: [{len(input_path_list)}]')

        # Get model path
        model_path = os.path.join('models', f'{model}.pth')
        print(f'Model path: [{model_path}]')
        if os.path.isfile(model_path):
            data_loader = _load_data(input_path_list)
            net = _define_model(model, model_path, gpu)
            # Inference for each image
            for i_test, data_test in enumerate(data_loader):
                # Log
                start_time = time.time()
                image_name = input_path_list[i_test].split("/")[-1]
                try:
                    # Prepare input
                    # noinspection PyUnresolvedReferences
                    inputs_test = data_test['image'].type(torch.FloatTensor)
                    if torch.cuda.is_available():
                        inputs_test = Variable(inputs_test.cuda())
                    else:
                        inputs_test = Variable(inputs_test)

                    # Inference
                    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

                    # Normalize
                    prediction = d1[:, 0, :, :]
                    prediction = _normalize_prediction(prediction)

                    # Save output
                    _save_output(input_path_list[i_test], prediction, output_path)

                    # Clean
                    del d1, d2, d3, d4, d5, d6, d7

                    # Log
                    total_time = time.time() - start_time
                    print(f'({i_test + 1}/{len(input_path_list)}): {image_name} - {total_time:.2f}s')

                except Exception as e:
                    print(f'({i_test + 1}/{len(input_path_list)}): Image [{image_name}] could not be processed: [{e}]')
                    print(e)
        else:
            print(f'Model path specified do not exist: [{model_path}]')
    else:
        print(f'Input path specified do not exist: [{input_path}]')


if __name__ == '__main__':
    args = _parse_args()
    run(args.input_path, args.output_path, args.model, args.gpu, args.show_user_warnings)
