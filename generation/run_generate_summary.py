import time
import subprocess
import sys

def run_script():
    try:
        subprocess.check_call(["python", "summary_generation.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing run.py: {e}")
        return False

if __name__ == "__main__":
    while True:
        success = run_script()
        if not success:
            print("Waiting for 1 minute before retrying...")
            ###### penai.error.RateLimitError: That model is currently overloaded with other requests. #######
            time.sleep(60)  # Sleep for 60 seconds (1 minute)
        else:
            sys.exit(0)
