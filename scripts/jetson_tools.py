import os

def modify_power_mode(mode):
  """
  Modifies the NVIDIA power mode using the nvpmodel utility.

  Args:
    mode: The power mode number (integer or string) to set.
  """
  mode_str = str(mode)
  print(f"Attempting to set NVIDIA power mode to {mode_str}...")

  # nvpmodel command to set the power mode
  command = f"nvpmodel -m {mode_str}"
  
  # Execute the command and capture the exit status
  # os.system returns the exit code of the command
  exit_code = os.system(command)
  
  if exit_code == 0:
    print(f"Successfully set power mode to {mode_str}.")
    
    # Optional: Verify by querying the current state
    current_status = os.popen("nvpmodel -q").read().strip()
    print(f"Current Status:\n{current_status}")
  else:
    print(f"Error: Failed to set power mode. (Exit code: {exit_code})")
    print("Please ensure:")
    print("1. You are running this script with root privileges (sudo).")
    print(f"2. Mode '{mode_str}' is a valid power mode for your specific Jetson model.")


def modify_gpu_freq(freq):
  """
  Modifies the GPU frequency to the specified value.

  Args:
    freq: The new GPU frequency in Hz.
  """

  # Get the model name using the requested shell command
  try:
    model = os.popen("tr -d '\\0' < /proc/device-tree/model").read().strip()
  except Exception as e:
    print(f"Error reading model: {e}")
    return

  # Define the location of the GPU frequency files for each model
  if "Jetson Nano" in model:
    path = '/sys/devices/57000000.gpu/devfreq/57000000.gpu'
  elif "Xavier NX" in model:
    path = '/sys/devices/17000000.gv11b/devfreq/17000000.gv11b'
  elif "AGX" in model:
    path = '/sys/devices/17000000.gv11b/devfreq/17000000.gv11b'
  else:
    print(f"Model not supported: {model}")
    return

  print(f"Model: {model}")

  # Read available frequencies
  try:
    with open(f'{path}/available_frequencies', 'r') as file:
      available_freqs = [int(f) for f in file.read().split()]
  except FileNotFoundError:
    print(f"Error: Could not find GPU frequency files at {path}.")
    return

  # Check if freq is within valid range
  if freq not in available_freqs:
    print(f"Error: Frequency {freq}Hz not supported. Valid options: {available_freqs}")
    return

  # Attempt to write the frequency to min and max files (with sudo)
  try:
    with open(f'{path}/min_freq', 'w') as min_file:
      min_file.write(str(freq))
    with open(f'{path}/max_freq', 'w') as max_file:
      max_file.write(str(freq))
    print(f"GPU frequency set to {freq}Hz.")
  except PermissionError:
    print("Error: Permission denied. Modifying GPU frequencies requires root privileges (run with sudo).")
  except OSError as e:
    print(f"Error setting GPU frequency: {e}")





def modify_variable(config_file, variable, seperator, value):
  """
  Modifies the variable value in config file.

  Args:
    config_file: The path to the config file.
    variable: The name of the variable to modify.
    seperator: The seperator between the variable and value.
    value: The new value.
  """
# Open the file in read mode
  with open(config_file, "r") as file:
    lines = file.readlines()

  # Modify line with GPU_FREQ declaration
  modified_lines = []
  for line in lines:
    if line.startswith(f"{variable} {seperator}"):
      # Extract the beginning part of the line
      start_of_line = line.split(seperator)[0]
      # Combine with freq to set the value
      modified_line = start_of_line + seperator+ ' ' + str(value) + '\n'
    else:
      modified_line = line

    modified_lines.append(modified_line)

  # Open the file again in write mode (overwrites existing content)
  with open(config_file, "w") as file:
    file.writelines(modified_lines)


def read_variable(file, variable, seperator):
  """
  Reads the value of a variable in a file.

  Args:
    file: The path to the config file.
    variable: The name of the variable to read.
  """
  # Open the file in read mode
  with open(file, "r") as file:
    lines = file.readlines()

  # Find the line with the variable
  for line in lines:
    if line.startswith(f"{variable} "):
      value = line.split(seperator)[1]
      # Remove any leading or trailing whitespace and also remove any comments
      return value.split("#")[0].strip()
    
  return None

