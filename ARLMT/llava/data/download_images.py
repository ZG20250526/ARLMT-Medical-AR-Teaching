# import os
# import json
# import shutil
# from tqdm import tqdm
# import tarfile
# import argparse
# from urllib.error import HTTPError
# import urllib.request
#
#
# def main(args):
#   input_data = []
#   with open(args.input_path) as f:
#     for line in f:
#       input_data.append(json.loads(line))
#
#   # Download all PMC articles
#   print('Downloading PMC articles')
#   for idx, sample in enumerate(tqdm(input_data)):
#     try:
#       urllib.request.urlretrieve(sample['pmc_tar_url'], os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url'])))
#     except HTTPError as e:
#       print('Error downloading PMC article: {}'.format(sample['pmc_tar_url']))
#       continue
#
#
#   # Untar all PMC articles
#   print('Untarring PMC articles')
#   for sample in tqdm(input_data):
#     fname = os.path.join(args.pmc_output_path, os.path.basename(os.path.join(sample['pmc_tar_url'])))
#     tar = tarfile.open(fname, "r:gz")
#     tar.extractall(args.pmc_output_path)
#     tar.close()
#
#   # Copy to images directory
#   print('Copying images')
#   for sample in tqdm(input_data):
#     src = os.path.join(args.pmc_output_path, sample['image_file_path'])
#     dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
#     shutil.copyfile(src, dst)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, default='/home/ubuntu/projects/LLaVA-Med/data/llava_med_image_urls.jsonl')
#     parser.add_argument('--pmc_output_path', type=str, default='/home/ubuntu/projects/LLaVA-Med/data/pmc_articles/')
#     parser.add_argument('--images_output_path', type=str, default='/home/ubuntu/projects/LLaVA-Med/data/images/')
#     args = parser.parse_args()
#     main(args)
import os
import json
import shutil
from tqdm import tqdm
import tarfile
import argparse
from urllib.error import HTTPError
import urllib.request
import hashlib

def check_file_integrity(file_path, expected_sha256=None):
   """Check if the file exists and its SHA-256 matches (if provided)."""
   if not os.path.exists(file_path):
       return False

   if expected_sha256:
       with open(file_path, 'rb') as f:
           file_sha256 = hashlib.sha256(f.read()).hexdigest()
           return file_sha256 == expected_sha256
   return True

def main(args):
   input_data = []
   with open(args.input_path) as f:
       for line in f:
           input_data.append(json.loads(line))

   # Download all PMC articles only if they don't exist or are incomplete
   print('Downloading PMC articles')
   for sample in tqdm(input_data):
       pmc_tar_url = sample['pmc_tar_url']
       local_tar_path = os.path.join(args.pmc_output_path, os.path.basename(pmc_tar_url))
       expected_sha256 = sample.get('sha256')  # Assuming the JSON contains the SHA-256 of the file

       if not check_file_integrity(local_tar_path, expected_sha256):
           try:
               urllib.request.urlretrieve(pmc_tar_url, local_tar_path)
           except HTTPError as e:
               print(f'Error downloading PMC article: {pmc_tar_url}')
               continue

   # Untar all PMC articles
   print('Untarring PMC articles')
   for sample in tqdm(input_data):
       local_tar_path = os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url']))
       tar = tarfile.open(local_tar_path, "r:gz")
       tar.extractall(args.pmc_output_path)
       tar.close()

   # Copy to images directory
   print('Copying images')
   for sample in tqdm(input_data):
       src = os.path.join(args.pmc_output_path, sample['image_file_path'])
       dst = os.path.join(args.images_output_path, sample['pair_id'] + '.jpg')
       if os.path.exists(src):
           shutil.copyfile(src, dst)
       else:
           print(f"Image file {src} does not exist, skipping.")

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_path', type=str, default='/media/ubuntu/12TB/projects/LLaVA-Med/data/llava_med_image_urls.jsonl')
   parser.add_argument('--pmc_output_path', type=str, default='/media/ubuntu/12TB/projects/LLaVA-Med/data/pmc_articles/')
   parser.add_argument('--images_output_path', type=str, default='/media/ubuntu/12TB/projects/LLaVA-Med/data/images/')
   args = parser.parse_args()
   main(args)