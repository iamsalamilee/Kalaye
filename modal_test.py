import modal

# 1. We name the app so it shows up neatly in your dashboard
app = modal.App("kalaye-gpu-test")

# 2. We tell Modal to turn on a giant T4 GPU for this specific function!
@app.function(gpu="T4")
def check_gpu():
    import subprocess
    print("🚀 [CLOUD] Booting up inside Modal's data center...")
    
    try:
        # We ask the server to print out its Nvidia GPU stats
        print("🚀 [CLOUD] Checking GPU hardware:")
        output = subprocess.check_output("nvidia-smi -L", shell=True)
        print(output.decode("utf-8"))
        return "GPU check passed!"
    except Exception as e:
        return f"Whoops, no GPU found: {e}"

# 3. This is what runs locally on your laptop
@app.local_entrypoint()
def main():
    print("💻 [LOCAL] Hey from your laptop! Sending request to the cloud...")
    
    # The magical ".remote()" command is what teleports the execution to the cloud
    result = check_gpu.remote()
    
    print(f"💻 [LOCAL] Cloud finished and returned: {result}")
