import subprocess
import sys

def run_command(command, step_name):
    print(f"\n========================================")
    print(f"{step_name}")
    print(f"========================================")
    
    try:
        # Run the command and stream the output to the console
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during {step_name}. Aborting update.")
        return False

def main():
    print("\n========================================")
    print("StockSense India - Automated Data Update")
    print("========================================")

    # Step 1: Train Models
    if not run_command("python batch_train.py", "Step 1/3: Training AI Models (This will take a few minutes)..."):
        sys.exit(1)

    # Step 2: Run Execution Pipeline
    if not run_command("python batch_runner.py", "Step 2/3: Running Execution Pipeline (Fetching data & predicting)..."):
        sys.exit(1)

    # Step 3: Git Operations
    print(f"\n========================================")
    print(f"Step 3/3: Committing and Pushing to GitHub...")
    print(f"========================================")
    
    try:
        # Add files
        subprocess.run("git add data/ models/", shell=True, check=True)
        
        # Commit (This will return a non-zero exit code if there's nothing to commit, which is fine)
        commit_result = subprocess.run('git commit -m "Automated local update: Refreshed AI models and stock data"', shell=True)
        
        if commit_result.returncode == 0:
            print("Changes committed. Pushing to GitHub...")
            # Push changes
            subprocess.run("git push origin master", shell=True, check=True)
            print("\n✅ Update Complete! Render will deploy the changes shortly.")
        else:
            print("\n⚠️ No new data changes to commit. Push skipped.")
            
    except subprocess.CalledProcessError as e:
         print(f"\n❌ Error during Git operations. Please check your connection or git status.")
         sys.exit(1)

if __name__ == "__main__":
    main()
