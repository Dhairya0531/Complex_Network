from PIL import Image, ImageDraw, ImageFont
import os

# Configuration: Metrics (Rows) x Cities (Cols)
CITIES = ["Bengaluru", "Berlin", "London", "Sydney"]
METRICS = [
    "Avg Queue", "Travel Time", "Throughput", "Wait Variance",
    "Throughput vs Demand", "Travel vs Demand", "Topology (BC)", "Control Params"
]

def create_final_grid():
    rows = 8
    cols = len(CITIES)
    
    # Open a sample image to get dimensions
    sample_path = f"{CITIES[0]}/plot_1.png"
    if not os.path.exists(sample_path):
        print(f"Error: {sample_path} not found. Make sure to run the simulations first.")
        return
        
    sample_img = Image.open(sample_path)
    w, h = sample_img.size
    
    # Margin for Row Labels (Left) and Column Labels (Top)
    left_margin = 1500
    top_margin = 600
    
    grid = Image.new('RGB', (w * cols + left_margin, h * rows + top_margin), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Font setup
    try:
        # Standard location for fonts on macOS
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 300)
        font_med = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 220)
    except:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
    
    for c_idx, label in enumerate(CITIES):
        # Draw Column Title (City Name) - Centered above the column
        title_x = c_idx * w + left_margin + w // 2
        draw.text((title_x, top_margin // 2), label, fill='black', font=font_large, anchor="mm")
        
        for r_idx in range(rows):
            # Draw Row Label (Metric Name) - Only for the first column
            if c_idx == 0:
                label_y = r_idx * h + top_margin + h // 2
                draw.text((20, label_y), METRICS[r_idx], fill='black', font=font_med, anchor="lm")
            
            # Load and paste plot
            plot_path = f"{label}/plot_{r_idx+1}.png"
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                grid.paste(img, (c_idx * w + left_margin, r_idx * h + top_margin))
            else:
                print(f"Warning: {plot_path} missing.")
            
    grid.save("final_paper_grid_hd.png", dpi=(300, 300))
    print("Successfully created final_paper_grid_hd.png with visible city names.")

if __name__ == "__main__":
    create_final_grid()
