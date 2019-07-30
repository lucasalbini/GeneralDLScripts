#!/bin/sh
uptime>time
cat time | $proc = awk '{print $NF}'
if ["$proc" -lt  1]; then 
	poweroff
fi
