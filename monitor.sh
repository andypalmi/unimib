#!/bin/bash

# Default interval value
interval=0.1

# Check if an argument is provided
if [ $# -gt 0 ]; then
  interval=$1
fi

# Run the watch command with the specified or default interval
watch -n "$interval" ./monitor_commands.sh