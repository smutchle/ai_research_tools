#!/bin/bash

# List of target processes to kill
targets=(
  "research_rag_bot_app.py"
)

echo "Searching for processes to terminate..."

# Flag to track if any processes were found
found=false

# Loop through each target process
for target in "${targets[@]}"; do
  # Find processes matching the target name
  pids=$(pgrep -f "$target")
  
  if [ -n "$pids" ]; then
    found=true
    echo "Found process(es) for $target with PID(s): $pids"
    
    # Kill each process
    for pid in $pids; do
      echo "Killing process $pid ($target)..."
      kill -9 $pid
      
      # Check if kill was successful
      if [ $? -eq 0 ]; then
        echo "✓ Successfully terminated process $pid"
      else
        echo "✗ Failed to terminate process $pid"
      fi
    done
  else
    echo "No processes found for $target"
  fi
done

if [ "$found" = false ]; then
  echo "No matching processes were found."
else
  echo "Process termination complete."
fi