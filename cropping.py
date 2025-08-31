import cv2
import os

def crop_image_into_patches(image_path, output_folder, patch_size=(300, 300), overlap=0):
    """
    Splits an image into patches of given size with optional overlap.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save patches.
        patch_size (tuple): (width, height) of each patch.
        overlap (int): Number of pixels to overlap between patches.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"⚠️ Could not load image at {image_path}")
    
    h, w, _ = image.shape
    pw, ph = patch_size

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    patch_id = 0
    for y in range(0, h, ph - overlap):
        for x in range(0, w, pw - overlap):
            # Ensure patch fits inside
            x_end = min(x + pw, w)
            y_end = min(y + ph, h)

            patch = image[y:y_end, x:x_end]

            # Save patch
            patch_filename = os.path.join(output_folder, f"patch5_{patch_id}.jpg")
            cv2.imwrite(patch_filename, patch)
            patch_id += 1

    print(f"✅ {patch_id} patches saved in {output_folder}")

# --- Run the function here ---
if __name__ == "__main__":
    image_path = "/home/vedanshi/Desktop/mussel_detection_app/raw_images/1.jpeg"
    out_path = "/home/vedanshi/Desktop/mussel_detection_app/raw_images/patch5"
    
    crop_image_into_patches(image_path, out_path, patch_size=(300, 300), overlap=100)

