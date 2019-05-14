# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import shutil

import torch
from PIL import Image

from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from data.base_dataset import get_params, get_transform

import runway
from runway.data_types import image


@runway.setup(options={'generator_checkpoint': runway.file(extension='.pth')})
def setup(opts):
    generator_checkpoint_path = opts['generator_checkpoint']
    try:
        os.makedirs('checkpoints/pretrained/')
    except OSError:
        pass
    shutil.copy(generator_checkpoint_path, 'checkpoints/pretrained/latest_net_G.pth')

    opt = TestOptions(args=['--name', 'pretrained',
                            '--netG', 'local',
                            '--ngf', '32',
                            '--resize_or_crop', 'none',
                            ]).parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)
    return {'model': model, 'opt': opt}


@runway.command(name='generate',
                inputs={ 'label_map': image(),
                         'instance_map': image() },
                outputs={ 'image': image() })
def generate(model, args):
    opt = model['opt']
    model = model['model']

    label_map = args['label_map']
    instance_map = args['instance_map']
    in_data = _get_input(label_map, instance_map, opt)

    generated = model.inference(in_data['label'], in_data['inst'], in_data['image'])

    im = util.tensor2im(generated.data[0])

    return {
        'image': im
    }


def _get_input(label_image, instance_image, opt):
    A = label_image
    params = get_params(opt, A.size)
    if opt.label_nc == 0:
      transform_A = get_transform(opt, params)
      A_tensor = transform_A(A.convert('RGB'))
    else:
      transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
      A_tensor = transform_A(A) * 255.0

    B_tensor = inst_tensor = feat_tensor = torch.tensor(0)

    if not opt.no_instance:
      inst = instance_image
      inst_tensor = transform_A(inst)

    input_dict = {'label': A_tensor.unsqueeze(0), 'inst': inst_tensor.unsqueeze(0),
                  'image': B_tensor, 'feat': feat_tensor, 'path': ''}

    return input_dict


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888)
