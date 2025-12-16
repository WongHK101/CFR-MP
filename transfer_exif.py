"""
python transfer_exif.py --src "/path/to/source_images" --dst "/path/to/mepma_results"
"""

import os
import argparse
import piexif
from PIL import Image, ExifTags

def create_tag_name_to_id_map():
    """
    Creates a reverse lookup dictionary from Tag Name to Tag ID.
    e.g., 'Make' -> 271, 'Model' -> 272
    """
    tag_map = {}
    for k, v in ExifTags.TAGS.items():
        tag_map[v] = k
    return tag_map

def transfer_exif(src_dir, dst_dir, ignore_tags=None):
    """
    Transfers EXIF data from source images to target images, filtering specified tags.
    """
    if ignore_tags is None:
        ignore_tags = []
    
    # Create the mapping for tag lookup
    name_to_id = create_tag_name_to_id_map()
    
    # Convert ignore list names to IDs
    ignore_ids = []
    for name in ignore_tags:
        if name in name_to_id:
            ignore_ids.append(name_to_id[name])
        else:
            print(f"Warning: Tag '{name}' not found in standard EXIF tags. Skipping.")

    # Get list of files in destination directory
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff')
    files = [f for f in os.listdir(dst_dir) if f.lower().endswith(supported_extensions)]
    
    print(f"Starting process...")
    print(f"Source Directory: {src_dir}")
    print(f"Target Directory: {dst_dir}")
    print(f"Tags to remove: {ignore_tags}")
    print(f"Found {len(files)} target files.")

    processed_count = 0
    
    for filename in files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"[Skip] Source file not found: {filename}")
            continue
            
        try:
            # 1. Read EXIF data from source (headers only, fast)
            src_img = Image.open(src_path)
            if 'exif' not in src_img.info:
                print(f"[Skip] No EXIF in source: {filename}")
                continue
            
            exif_dict = piexif.load(src_img.info['exif'])
            
            # 2. Iterate and remove specified tags
            for ifd_name in exif_dict:
                if ifd_name == "thumbnail": 
                    continue 
                
                # Use list(keys) to allow modification during iteration
                for tag_id in list(exif_dict[ifd_name].keys()):
                    if tag_id in ignore_ids:
                        del exif_dict[ifd_name][tag_id]

            # 3. Dump modified EXIF to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # 4. Insert into target file (lossless operation)
            piexif.insert(exif_bytes, dst_path)
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} images...")

        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

    print(f"Done! Successfully transferred EXIF for {processed_count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer EXIF data between folders while filtering specific tags.")
    
    parser.add_argument(
        "--src", 
        required=True, 
        help="Path to the source folder containing images with correct EXIF."
    )
    
    parser.add_argument(
        "--dst", 
        required=True, 
        help="Path to the destination folder (e.g., ME-PMA results) to write EXIF to."
    )
    
    parser.add_argument(
        "--ignore", 
        nargs="+", 
        default=["Make", "Model"], 
        help="List of EXIF tag names to remove/ignore (default: Make Model)."
    )

    args = parser.parse_args()
    
    if not os.path.isdir(args.src):
        print(f"Error: Source directory '{args.src}' does not exist.")
        exit(1)
    if not os.path.isdir(args.dst):
        print(f"Error: Target directory '{args.dst}' does not exist.")
        exit(1)

    transfer_exif(args.src, args.dst, args.ignore)