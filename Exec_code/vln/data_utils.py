import os
import json
from ipdb import set_trace

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if 'sample' in split:
            filepath = os.path.join(anno_dir, split)
            with open(filepath) as f:
                new_data = json.load(f)

        elif "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s.json' % (split))
                print(filepath)
            else:
                raise NotImplementedError('unsupported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

        # Join
        data += new_data
    return data


def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                obj2vps.setdefault(scan+'_'+objid, [])
                obj2vps[scan+'_'+objid].append(vp)
    return obj2vps