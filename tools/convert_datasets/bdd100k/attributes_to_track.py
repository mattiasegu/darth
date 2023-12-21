"""Add domain attributes to the BDD100K tracking annotations in the Scalabel
format."""
import argparse
import os
import os.path as osp
from scalabel.label.io import load


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add domain attributes to the BDD100K tracking annotations.'
    )
    parser.add_argument(
        '-n', 
        '--num_processes',
        type=int,
        default=0,
        help='number of processes to load the Scalabel annotations.',
    )
    parser.add_argument(
        '-d', 
        '--detection_json_dir',
        type=str,
        default='data/bdd100k/labels/det_20/',
        help='path to directory containing json files with detection annotations.',
    )
    parser.add_argument(
        '-t', 
        '--tracking_json_dir',
        type=str,
        default='data/bdd100k/labels/box_track_20/',
        help='path to directory containing json files with tracking annotations.',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='data/bdd100k/labels/box_track_20_with_domains/',
        help='path to directory where to save the tracking labels with domain '
        'annotations.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for split in ['train', 'val']:
        track_dir = os.path.join(args.tracking_json_dir, split)
        det_file = os.path.join(args.detection_json_dir, 
                                'det_' + split + '.json')
        assert os.path.exists(track_dir)
        assert os.path.exists(det_file)

        split_output_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(split_output_dir):
            os.mkdir(split_output_dir)

        det_anns = load(
            det_file,
            nprocs=args.num_processes,
        )

        # Create dictionary with names and indexes of detection frames
        det_anns_dict = {}
        for i, f in enumerate(det_anns.frames):
            det_anns_dict[f.name] = f.attributes

        # Parse domain annotations from detection set and assign them to the 
        # corresponding frames in the tracking set
        dir_len = len(os.listdir(track_dir))
        for i, track_file in enumerate(os.listdir(track_dir)):
            if track_file.endswith('.json'):
                track_anns = load(
                    os.path.join(track_dir, track_file),
                    nprocs=args.num_processes,
                )
                
                for j, track_frame in enumerate(track_anns.frames):
                    try:
                        str_list = track_frame.name.split('-')
                        det_frame_name = '.'.join(['-'.join(str_list[0:2]),
                                                str_list[2].split('.')[-1]])
                        det_attributes = det_anns_dict[det_frame_name]
                        track_anns.frames[j].attributes = det_attributes
                    except:
                        print(f"{det_frame_name} not found.")

                save_path = os.path.join(split_output_dir, track_file)

                if os.path.exists(save_path):
                    print(f'Overwriting file {i} of {dir_len}: {save_path}')
                else:
                    print(f'Writing file {i} of {dir_len}: {save_path}')

                with open(save_path, "w") as f:
                    f.write(track_anns.json())

        print(f'Finished writing {split} split annotations with domain labels')


if __name__ == "__main__":
    main()
