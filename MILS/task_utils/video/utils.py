# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from task_utils.video.simple_tokenizer import SimpleTokenizer as _Tokenizer
from task_utils.video.viclip import ViCLIP
import torch
import numpy as np
import cv2


clip_candidates = {'viclip':None, 'clip':None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def _frame_from_video(video):
    if not video.isOpened():
        raise ValueError()
    
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


def normalize(data):
    return (data/255.0-v_mean)/v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum), (len(vid_list), fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def retrieve_text(frames, texts, name='viclip', topk=5, device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]