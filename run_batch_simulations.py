import subprocess
import os
import sys
import shutil

# List of cities to simulate (mapping to folder names)
CITIES = [
    "Bengaluru, India",
    "Berlin, Germany",
    "Sydney, Australia",
    "London, United Kingdom"
]

def run_simulation(city):
    # Create folder name from city name
    folder_name = city.split(",")[0].replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    print(f"\n>>> Starting simulation for: {city}")
    
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Modify the script's configuration to point to the current city
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
        # Detect OS and set correct venv python path
        if sys.platform == "win32":
            python_exe = os.path.join(".venv", "Scripts", "python.exe")
        else:
            python_exe = os.path.join(".venv", "bin", "python")
        
        # If venv doesn't exist, fall back to sys.executable
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        subprocess.run([python_exe, temp_script], check=True)
        
        # Move generated files to the city folder
        # The script generates plot*.png, plot*.html, and results_summary_*.txt
        files_to_move = [f for f in os.listdir(".") if f.endswith(".png") or f.endswith(".html") or f.startswith("results_summary_")]
        for file in files_to_move:
            # We move plot images and the specific summary for this city
            if file.startswith("plot") or (file.startswith("results_summary_") and folder_name in file):
                dest_path = os.path.join(folder_name, file)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(file, dest_path)
                
        print(f">>> Completed simulation for: {city}. Results in ./{folder_name}/")
    except subprocess.CalledProcessError as e:
        print(f">>> FAILED simulation for: {city}. Exit code: {e.returncode}")
    except Exception as e:
        print(f">>> ERROR for {city}: {str(e)}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    for city in CITIES:
        run_simulation(city)
