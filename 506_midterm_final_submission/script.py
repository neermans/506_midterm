import subprocess

def run_script(file_name):
    """Function to run a Python script using subprocess and handle output."""
    try:
        # Run the script
        result = subprocess.run(['python3', file_name], check=True, text=True, capture_output=True)
        # Print the output from the script (stdout)
        print(f"Output from {file_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        # Print the error output from the script (stderr) if it fails
        print(f"Error running {file_name}:\n{e.stderr}")

def main():
    # List of scripts to run
    scripts = ['pre_process_datafile.py', 'xgboost_implementation.py']
    
    # Loop through and run each script
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)

if __name__ == "__main__":
    main()
