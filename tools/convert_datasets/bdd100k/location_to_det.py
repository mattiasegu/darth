"""Add location attribute to the BDD100K detection annotations in the Scalabel
format."""

import argparse
import json
import os
import os.path as osp
from time import sleep

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from multiprocessing import cpu_count
from scalabel.label.io import load


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add location attributes to the BDD100K detection annotations.'
    )
    parser.add_argument(
        '-n', 
        '--num_processes',
        type=int,
        default=0,
        help='number of processes to load the Scalabel annotations.',
    )
    parser.add_argument(
        '-i',
        "--info_json_dir",
        type=str,
        default="data/bdd100k/labels/info/100k/",
        help='path to directory containing json files with location annotations.',
    )
    parser.add_argument(
        '-d',
        "--detection_json_dir",
        type=str,
        default="data/bdd100k/labels/det_20/",
        help='path to directory containing json files with detection annotations.',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='data/bdd100k/labels/det_20_with_locations/',
        help='path to directory where to save the detection labels with domain '
        'annotations.',
    )
    return parser.parse_args()
    

def get_location_from_latitude_longitude(latitude, longitude):

    geolocator = Nominatim(user_agent="application")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    location = reverse((latitude, longitude), language='en', exactly_one=True)
    
    # initialize Nominatim API
    # geolocator = Nominatim(user_agent="geoapi")    
    # location = geolocator.reverse(str(latitude)+","+str(longitude))
    address = location.raw['address']
    
    return address


def main():
    args = parse_args()

    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print(
        "Since Nominatum is rate limited, we can only make one request per "
        "second.")

    for split in ['train', 'val']:
        info_dir = osp.join(args.info_json_dir, split)
        det_file = osp.join(args.detection_json_dir, 
                                'det_' + split + '.json')
        assert osp.exists(info_dir)
        assert osp.exists(det_file)

        # Making detection output directory
        split_detection_output_dir = args.output_dir

        det_anns = load(
            det_file,
            nprocs=args.num_processes,
        )

        # Create dictionary with names and indexes of detection frames
        count_dict = {'city': {}, 'state': {}}
        domain_anns_dict = {}
        len_det_frames = len(det_anns.frames)
        for i, f in enumerate(det_anns.frames):
            print(f'Retrieving location of frame {i} of {len_det_frames}: {f.name}')
            sleep(1)
            video_name = f.name.split('.')[0]
            info_json_path = osp.join(info_dir, video_name + '.json')
            if os.stat(info_json_path).st_size:
                info_json = open(info_json_path)
                info_anns = json.load(info_json)
                info_json.close()

                # Derive frame location
                if 'locations' in info_anns and len(info_anns['locations']) > 0:
                    video_location_sample = info_anns['locations'][0]
                    latitude = video_location_sample['latitude']
                    longitude = video_location_sample['longitude']

                    location_dict = get_location_from_latitude_longitude(
                        latitude, longitude)
                    city = location_dict.get('city', 'undefined')
                    state = location_dict.get('state', 'undefined')
                else:
                    print(f'Locations not available for {info_json_path}')
                    city = 'undefined'
                    state = 'undefined'
            else:
                print(f'Empty json {info_json_path}')
                city = 'undefined'
                state = 'undefined'

            if city in count_dict['city']:
                count_dict['city'][city] += 1
            else:
                count_dict['city'][city] = 1

            if state in count_dict['state']:
                count_dict['state'][state] += 1
            else:
                count_dict['state'][state] = 1
                
            #  Store domain attributes in dictionary
            domain_attributes = f.attributes
            domain_attributes['city'] = city
            domain_attributes['state'] = state
            domain_anns_dict[f.name] = domain_attributes
            det_anns.frames[i].attributes = domain_attributes

        save_path = osp.join(split_detection_output_dir,
                                 det_file.split('/')[-1])
        if osp.exists(save_path):
            print(f'Overwriting file {save_path}')
        else:
            print(f'Writing file {save_path}')

        with open(save_path, "w") as f:
            f.write(det_anns.json())

        print(f'Finished writing det {split} annotations with domain labels')

        # Log city and state distribution
        print('City distribution: ')
        for city in count_dict['city']:
            print(city, count_dict['city'][city])
        print('State distribution: ')
        for state in count_dict['state']:
            print(state, count_dict['state'][state])
         

if __name__ == "__main__":
    main()
