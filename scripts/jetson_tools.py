import os

def modify_gpu_freq(freq):
  """
  Modifies the GPU frequency to the specified value.

  Args:
    freq: The new GPU frequency in Hz.
  """

  # Get the model name 
  model = os.popen("cd /home/iloudaros/LoudVA && make model").read().strip()

  # Define the location of the GPU frequency files for each model
  if model == "NVIDIA Jetson Nano Developer Kit":
    path = '/sys/devices/57000000.gpu/devfreq/57000000.gpu'
  elif model == "NVIDIA Jetson Xavier NX Developer Kit":
    path = '/sys/devices/17000000.gv11b/devfreq/17000000.gv11b'
  elif model == "Jetson-AGX":
    path = '/sys/devices/17000000.gv11b/devfreq/17000000.gv11b'
  else:
    print('Model not supported')

  print(f"Model: {model}")

  # Read available frequencies
  with open(f'{path}/available_frequencies', 'r') as file:
    available_freqs = [int(f) for f in file.read().split()]

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

