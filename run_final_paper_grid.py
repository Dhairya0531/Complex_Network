import subprocess
import os
import sys
import shutil

# Selected cities for the paper grid
CITIES = [
    "Bengaluru, India",
    "Berlin, Germany",
    "London, United Kingdom",
    "Sydney, Australia"
]

def run_simulation(city):
    folder_name = city.split(",")[0].replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    print(f"\n>>> Starting simulation for: {city}")
    
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith('PLACE = '):
            new_lines.append(f'PLACE = "{city}"\n')
        else:
            new_lines.append(line)
            
    temp_script = f"temp_main_{folder_name}.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    try:
        python_exe = os.path.join(".venv", "bin", "python")
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        subprocess.run([python_exe, temp_script], check=True)
        
        # Move generated files (plot_1.png to plot_8.png)
        for i in range(1, 9):
            filename = f"plot_{i}.png"
            if os.path.exists(filename):
                dest_path = os.path.join(folder_name, filename)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(filename, dest_path)
        
        # Move summary
        summary_file = f"results_summary_{folder_name}.txt"
        if os.path.exists(summary_file):
            shutil.move(summary_file, os.path.join(folder_name, summary_file))
                
        print(f">>> Completed simulation for: {city}. Results in ./{folder_name}/")
    except Exception as e:
        print(f">>> ERROR for {city}: {str(e)}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    for city in CITIES:
        run_simulation(city)
    
    print("\n>>> All simulations complete. Generating final grid...")
    python_exe = os.path.join(".venv", "bin", "python")
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    subprocess.run([python_exe, "paper_grid_generator.py"], check=True)
