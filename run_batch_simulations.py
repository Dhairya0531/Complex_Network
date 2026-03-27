import subprocess
import os
import sys
import shutil

# List of cities to simulate
CITIES = [
    "Nancy, France",
    "Chandigarh, India",
    "Berlin, Germany",
    "Sydney, Australia",
    "Bengaluru, India",
    "Mumbai, Maharashtra, India",
    "Delhi, India"
]

def run_simulation(city):
    # Create folder name from city name
    folder_name = city.split(",")[0].replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    print(f"\n>>> Starting simulation for: {city}")
    
    # Define the command to run main.py with the specific city
    # We use a small wrapper or environment variable if main.py supported it, 
    # but since I shouldn't change the code, I'll use a temporary copy or 
    # simply edit the PLACE line in a copy.
    
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Modify the copy's configuration
    new_lines = []
    for line in lines:
        if line.startswith('PLACE = '):
            new_lines.append(f'PLACE = "{city}"\n')
        elif 'fig6.legend' in line:
            # Vertical legend on the right
            new_line = 'fig6.legend(handles, labels_leg, loc="center left", ncol=1, bbox_to_anchor=(1.0, 0.5), prop={"size": 18})\n'
            new_lines.append(new_line)
        elif 'plt.tight_layout(rect=[0, 0, 1, 0.88])' in line:
            # Remove top margin since legend moved to the right
            new_lines.append('plt.tight_layout()\n')
        elif 'ax1.set_ylim(0, max(values) * 1.2)' in line:
            new_lines.append('clean_vals = [v for v in values if not np.isnan(v)]\n')
            new_lines.append('if clean_vals: ax1.set_ylim(0, max(clean_vals) * 1.2)\n')
        elif 'ax2.set_ylim(0, max(values) * 1.2)' in line:
            new_lines.append('clean_vals = [v for v in values if not np.isnan(v)]\n')
            new_lines.append('if clean_vals: ax2.set_ylim(0, max(clean_vals) * 1.2)\n')
        elif 'ax3.set_ylim(0, max(values) * 1.2)' in line:
            new_lines.append('clean_vals = [v for v in values if not np.isnan(v)]\n')
            new_lines.append('if clean_vals: ax3.set_ylim(0, max(clean_vals) * 1.2)\n')
        else:
            new_lines.append(line)
            
    temp_script = f"temp_main_{folder_name}.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    try:
        # Run the modified script using the venv python
        python_exe = os.path.join(".venv", "Scripts", "python.exe")
        subprocess.run([python_exe, temp_script], check=True)
        
        # Move generated files to the city folder
        files_to_move = [f for f in os.listdir(".") if f.endswith(".png") or f.endswith(".html") or f.startswith("results_summary_")]
        for file in files_to_move:
            # Don't move the script itself or other cities' summaries if they exist
            if file.startswith("plot") or file == f"results_summary_{folder_name}.txt":
                dest_path = os.path.join(folder_name, file)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(file, dest_path)
                
        print(f">>> Completed simulation for: {city}. Results in ./{folder_name}/")
    except subprocess.CalledProcessError as e:
        print(f">>> FAILED simulation for: {city}. Skipping to next...")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    for city in CITIES:
        run_simulation(city)
