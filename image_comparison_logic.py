import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def calculate_ssim_full_image(image_path1, image_path2):
    """
    Calculates the Structural Similarity Index (SSIM) between two full images.
    Returns the SSIM score and a matplotlib Figure object.
    """
    try:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        if img1 is None or img2 is None:
            raise FileNotFoundError(
                f"Could not load one or both images: {image_path1}, {image_path2}"
            )

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Ensure images have the same dimensions for SSIM
        if gray1.shape != gray2.shape:
            print(
                f"Warning: Images have different dimensions ({gray1.shape} vs {gray2.shape}). Resizing the second image."
            )
            h, w = gray1.shape
            gray2 = cv2.resize(gray2, (w, h))
            img2 = cv2.resize(img2, (w, h)) # Also resize the color image for consistent display

        (score, diff) = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")

        # Create the matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes.ravel()

        ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Image 1")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Image 2")
        ax[1].axis("off")

        ax[2].imshow(diff, cmap=plt.cm.gray)
        ax[2].set_title(f"Difference Map (SSIM: {score:.4f})")
        ax[2].axis("off")

        plt.tight_layout()
        return score, fig

    except Exception as e:
        print(f"An error occurred during full image SSIM calculation: {e}")
        return None, None


def calculate_ssim_region(image_path1, image_path2, region_coords):
    """
    Calculates the Structural Similarity Index (SSIM) between specific regions of two images.
    Returns the SSIM score and a matplotlib Figure object.
    """
    try:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        if img1 is None or img2 is None:
            raise FileNotFoundError(
                f"Could not load one or both images: {image_path1}, {image_path2}"
            )
        
        # Unpack the single region_coords tuple
        x, y, w, h = region_coords

        # Safely extract regions, ensuring they are within bounds
        h1, w1, _ = img1.shape
        x_start1, y_start1 = max(0, x), max(0, y)
        x_end1, y_end1 = min(w1, x + w), min(h1, y + h)
        region1_color = img1[y_start1:y_end1, x_start1:x_end1]

        h2, w2, _ = img2.shape
        x_start2, y_start2 = max(0, x), max(0, y)
        x_end2, y_end2 = min(w2, x + w), min(h2, y + h)
        region2_color = img2[y_start2:y_end2, x_start2:x_end2]

        if region1_color.size == 0 or region2_color.size == 0:
            raise ValueError(
                "Extracted region is empty. Check coordinates and image dimensions."
            )

        # Convert to grayscale for comparison
        region1_gray = cv2.cvtColor(region1_color, cv2.COLOR_BGR2GRAY)
        region2_gray = cv2.cvtColor(region2_color, cv2.COLOR_BGR2GRAY)

        # Resize the second region to match the first if shapes differ
        if region1_gray.shape != region2_gray.shape:
            print(
                f"Warning: Region dimensions {region1_gray.shape} vs {region2_gray.shape}. Resizing Region 2."
            )
            target_h, target_w = region1_gray.shape
            region2_gray = cv2.resize(region2_gray, (target_w, target_h))
            region2_color = cv2.resize(region2_color, (target_w, target_h))

        (score, diff_map_region) = ssim(region1_gray, region2_gray, full=True)
        diff_map_region = (diff_map_region * 255).astype("uint8")

        # Create the matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes.ravel()
        
        ax[0].imshow(cv2.cvtColor(region1_color, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Image 1 Region")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(region2_color, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Image 2 Region")
        ax[1].axis("off")

        ax[2].imshow(diff_map_region, cmap=plt.cm.gray)
        ax[2].set_title(f"Region Diff Map (SSIM: {score:.4f})")
        ax[2].axis("off")

        plt.tight_layout()
        return score, fig

    except Exception as e:
        print(f"An error occurred during region SSIM calculation: {e}")
        return None, None