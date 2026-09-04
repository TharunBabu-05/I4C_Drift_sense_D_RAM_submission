import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def visualize(reference_path, search_path, pred_x, pred_y, gt_x=None, gt_y=None, output_path="visualization.png"):
    ref_img = np.array(Image.open(reference_path))
    search_img = np.array(Image.open(search_path))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot Reference
    ax1.imshow(ref_img, cmap='gray')
    ax1.set_title("Reference Pattern (100x)")
    ax1.axis('off')
    
    # Plot Search
    ax2.imshow(search_img, cmap='gray')
    ax2.set_title("Search Image (10x)")
    ax2.axis('off')
    
    # Ref size in search image is ~100x100
    ref_w, ref_h = 100, 100
    
    # Draw Prediction
    rect_pred = patches.Rectangle((pred_x - ref_w/2, pred_y - ref_h/2), ref_w, ref_h, 
                                  linewidth=2, edgecolor='r', facecolor='none', label='Prediction')
    ax2.add_patch(rect_pred)
    ax2.plot(pred_x, pred_y, 'rx', markersize=10)
    
    # Draw Ground Truth if available
    if gt_x is not None and gt_y is not None:
        rect_gt = patches.Rectangle((gt_x - ref_w/2, gt_y - ref_h/2), ref_w, ref_h, 
                                    linewidth=2, edgecolor='g', facecolor='none', linestyle='--', label='Ground Truth')
        ax2.add_patch(rect_gt)
        ax2.plot(gt_x, gt_y, 'g+', markersize=10)
        
        # Draw error vector
        ax2.plot([gt_x, pred_x], [gt_y, pred_y], 'y-', linewidth=1.5)
        
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--pred_x", type=float, required=True)
    parser.add_argument("--pred_y", type=float, required=True)
    parser.add_argument("--gt_x", type=float, default=None)
    parser.add_argument("--gt_y", type=float, default=None)
    parser.add_argument("--output", type=str, default="visualization.png")
    args = parser.parse_args()
    
    visualize(args.reference, args.search, args.pred_x, args.pred_y, args.gt_x, args.gt_y, args.output)

if __name__ == "__main__":
    main()
