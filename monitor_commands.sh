#!/bin/bash

# Query GPU information, replace headers, and align columns
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw --format=csv,noheader,nounits \
| awk 'BEGIN{print "VRAM: Used,VRAM: Total,GPU Usage %,Temp °C,Fan %,Draw W"} {print $0}' \
| column -t -s ","

# Print a newline for readability
echo

# Query memory information excluding swap, keeping headers, and align columns
free --giga | awk 'NR==1 || /^Mem:/' | sed 's/Mem://' | column -t