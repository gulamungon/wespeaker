# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, io, tarfile, time 
from pathlib import Path
import numpy as np
from pprint import pformat
import fire
import yaml
from torch.utils.data import DataLoader

from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, spk2id
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.executor import run_epoch
from wespeaker.dataset.dataset import Dataset

from tools.make_shard_list import write_tar_file



#from torch import tensor

def prep_archives(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    logger = get_logger(configs['exp_dir'], 'prep_archives.log')
    logger.info("exp_dir is: {}".format(configs['exp_dir']))
    logger.info("<== Passed Arguments ==>")
    # Print arguments into logs
    for line in pformat(configs).split('\n'):
        logger.info(line)

    # seed
    #set_seed(configs['seed'] + rank)
    set_seed(configs['seed'] )

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    print(len(spk2id_dict))
    print(max(spk2id_dict.values()))
    #sys.exit()
    #if rank == 0:
    logger.info("<== Data statistics ==>")
    logger.info("train data num: {}, spk num: {}".format(
        len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args']) # Adding False here
    batch_size = configs['dataloader_args']['batch_size']
    logger.info("<== Dataloaders ==>")
    logger.info("train dataloaders created")
    
    start_epoch = 1 # Add as input argument ?

    
    num_epochs = 38
    ark_size = 2**17
    #lab_all = np.zeros(0)
    lab_all = np.zeros( ark_size + configs['dataloader_args']['batch_size'] -1) # We will not reach bigger than this since after each batch we check if 
                                                                              # We have enough examples for the archive.
    #key_all = []
    key_all = [None]*(ark_size + configs['dataloader_args']['batch_size']-1)
    #feat_all = np.zeros((0, configs['dataset_args']['num_frms'], configs['dataset_args']['fbank_args']['num_mel_bins'] ) )
    feat_all = np.zeros((ark_size + configs['dataloader_args']['batch_size']-1, 
                         configs['dataset_args']['num_frms'], configs['dataset_args']['fbank_args']['num_mel_bins']))

    n_examples_ready = 0

    prefix = "shards"
    shards_dir = "/mnt/scratch/tmp/rohdin/wespeaker/ark/"
    shard_idx = 0
    sanity_dir = "/mnt/scratch/tmp/rohdin/wespeaker/sanity/"


    ts = time.time()

    #tmp_virt_f = tempfile.SpooledTemporaryFile( max_size = 1e9 ) # Alternative to io.BytesIO()
    #buf        = io.BytesIO()
    with io.BytesIO() as buf:

        for epoch in range(start_epoch, num_epochs + 1):
            train_dataset.set_epoch(epoch)
            for i, batch in enumerate(train_dataloader):



                key_tmp = batch['key']
                n_examples_this_batch = len(key_tmp)
                #key_all = key_all + key_tmp 
                key_all[ n_examples_ready:n_examples_ready + n_examples_this_batch ] = key_tmp

                lab_tmp = batch['label']
                lab_tmp = lab_tmp.numpy()
                #print(lab_tmp.shape)


                feat_tmp = batch['feat']
                feat_tmp = feat_tmp.numpy()
                #print(feat_tmp.shape)
                #feat_all = np.vstack( (feat_all, feat_tmp) )
                feat_all[ n_examples_ready:n_examples_ready + n_examples_this_batch, :, : ] = feat_tmp

                #lab_all = np.hstack( (lab_all, lab_tmp) )
                lab_all[ n_examples_ready:n_examples_ready + n_examples_this_batch] = lab_tmp

                n_examples_ready += n_examples_this_batch 
                #assert( len(key_all) == len(lab_all) == len(feat_all) )
                #logger.info("Epoch {}, Batch {}, Examples ready: {}".format(epoch, i, len(key_all) ) )
                logger.info("Epoch {}, Batch {}, Examples ready: {}".format(epoch, i, n_examples_ready ))
                

                #if len(key_all) >= ark_size:
                if n_examples_ready >= ark_size:
                    generate_time = time.time() -ts
                    #assert( len(key_all) == len(lab_all) == len(feat_all) )
                    #len_before = len(key_all)

                    tar_file = os.path.join(shards_dir, '{}_{:09d}.tar'.format(prefix, shard_idx))
                    ts = time.time()
                    with tarfile.open(tar_file, "w") as tar:
                        for n in range(ark_size):
                            #spk  = str( spk_all[ n ] )
                            lab  = lab_all[n]
                            feat = feat_all[ n, :, : ]
                            key  = key_all[ n ]

                            ts = time.time()

                            """
                            assert isinstance(spk, str)
                            spk_file = key + '.spk'
                            spk = spk.encode('utf8')
                            spk_data = io.BytesIO(spk)
                            spk_info = tarfile.TarInfo(spk_file)
                            spk_info.size = len(spk)
                            tar.addfile(spk_info, spk_data)
                            """

                            feat_file = key + '.feat.npy'
                            feat_info = tarfile.TarInfo(feat_file)

                            np.save(buf, feat)
                            feat_info.size = buf.tell()
                            buf.seek(0)
                            tar.addfile(feat_info, buf)
                            buf.seek(0)

                            lab_file = key + '.lab.npy'
                            lab_info = tarfile.TarInfo(lab_file)

                            np.save(buf, lab)
                            lab_info.size = buf.tell()
                            buf.seek(0)
                            tar.addfile(lab_info, buf)
                            buf.seek(0)

                            if n == 3:
                                Path( sanity_dir + "/" + lab_file ).parent.mkdir(parents=True, exist_ok=True)
                                np.save(sanity_dir + "/" + feat_file, feat)
                                np.save(sanity_dir + "/" + lab_file, lab)

                    #lab_all  = lab_all[ ark_size: ] 
                    #feat_all = feat_all[ ark_size:, :, : ]
                    #key_all  = key_all[ ark_size: ]

                    n_remaining = n_examples_ready - ark_size

                    lab_all[:n_remaining]        = lab_all[ ark_size : ark_size + n_remaining ] 
                    key_all[:n_remaining]        = key_all[ ark_size : ark_size + n_remaining ] 
                    feat_all[:n_remaining, :, :] = feat_all[ ark_size : ark_size + n_remaining, : ,: ] 

                    lab_all[n_remaining:]        = 0 
                    key_all[n_remaining:]        = [None]*(ark_size + configs['dataloader_args']['batch_size']-1 -n_remaining)
                    feat_all[n_remaining:, :, :] = 0 

                    #assert( len(key_all) == len(lab_all) == len(feat_all) )
                    #assert( (len_before - len(key_all)) == ark_size ) 

                    n_examples_ready = n_remaining  
                    write_time = time.time() - ts
                    logger.info('Stored archive {}. Generate time: {}. Write time: {}.'.format(tar_file, generate_time, write_time))
                    shard_idx += 1
                    ts = time.time()


if __name__ == '__main__':
    fire.Fire(prep_archives)
