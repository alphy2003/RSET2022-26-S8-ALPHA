import os
import cv2
import numpy as np
from tqdm import tqdm

ALIGNED_DIR = "aligned_images_128"    # input (aligned images)
MASK_DIR = "aligned_eye_masks"                # masks output folder (may mirror subfolders)
OUT_DIR = "masked_inputs"         # where masked images will be written

os.makedirs(OUT_DIR, exist_ok=True)

# Helper: possible mask filename patterns (in order of preference)
def possible_mask_paths(mask_dir, rel_dir, filename):
    """
    Given mask_dir, relative directory and image filename (xxx.jpg),
    generate plausible mask paths to check.
    """
    name_no_ext = os.path.splitext(filename)[0]
    candidates = [
        os.path.join(mask_dir, rel_dir, name_no_ext + "_mask.png"),
        os.path.join(mask_dir, rel_dir, name_no_ext + ".png"),
        os.path.join(mask_dir, rel_dir, name_no_ext + ".jpg"),
        os.path.join(mask_dir, rel_dir, name_no_ext + "_mask.jpg"),
    ]
    # also try mask folder without subdir (in case masks are flat)
    candidates += [
        os.path.join(mask_dir, name_no_ext + "_mask.png"),
        os.path.join(mask_dir, name_no_ext + ".png"),
        os.path.join(mask_dir, name_no_ext + ".jpg"),
    ]
    return candidates

missing_masks = []
copied_count = 0
err_count = 0
total_images = 0

print("Scanning aligned images and generating masked inputs...\n")

# Walk aligned dir so script handles both flat and nested structures
for root, dirs, files in os.walk(ALIGNED_DIR):
    # compute relative directory w.r.t ALIGNED_DIR
    rel_dir = os.path.relpath(root, ALIGNED_DIR)
    if rel_dir == ".":
        rel_dir = ""  # top-level
    
    # create corresponding output dir
    out_subdir = os.path.join(OUT_DIR, rel_dir) if rel_dir != "" else OUT_DIR
    os.makedirs(out_subdir, exist_ok=True)
    
    # iterate image files
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fname in tqdm(image_files, desc=f"Processing {rel_dir or 'root'}", leave=False):
        total_images += 1
        aligned_path = os.path.join(root, fname)
        
        # find mask path by trying candidates
        candidates = possible_mask_paths(MASK_DIR, rel_dir, fname)
        mask_path = None
        for c in candidates:
            if os.path.exists(c):
                mask_path = c
                break
        
        if mask_path is None:
            missing_masks.append(os.path.join(rel_dir, fname) if rel_dir else fname)
            continue
        
        # load files
        img = cv2.imread(aligned_path)
        if img is None:
            err_count += 1
            print(f"ERROR: cannot read image {aligned_path}")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            err_count += 1
            print(f"ERROR: cannot read mask {mask_path} (for image {aligned_path})")
            continue
        
        # normalize mask to binary 0/1
        try:
            binary_mask = (mask > 127).astype(np.uint8)
            # apply mask: set masked regions to black
            masked_img = img.copy()
            masked_img[binary_mask == 1] = 0

            # save masked image using same relative path; suffix _masked.jpg
            out_name = os.path.splitext(fname)[0] + "_masked.jpg"
            out_path = os.path.join(out_subdir, out_name)
            success = cv2.imwrite(out_path, masked_img)
            if not success:
                err_count += 1
                print(f"ERROR: failed to write {out_path}")
            else:
                copied_count += 1
        except Exception as e:
            err_count += 1
            print(f"EXCEPTION processing {aligned_path} with mask {mask_path}: {e}")

# Summary
print("\n=== Summary ===")
print("Total aligned images encountered :", total_images)
print("Masked images created              :", copied_count)
print("Images missing masks               :", len(missing_masks))
print("Other errors                       :", err_count)

# write missing list
with open("missing_masks_check.txt", "w") as f:
    for x in missing_masks:
        f.write(x + "\n")

print("Missing mask list saved to missing_masks_check.txt")
