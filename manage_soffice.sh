#!/bin/bash
# Configuration
MAX_MEM_PERCENT=15          # If total memory usage exceeds 15%
CPU_USAGE_THRESHOLD=5         # and instantaneous CPU usage is below 5%
CHECK_INTERVAL=30             # Check every 30 seconds
IDLE_COUNT_THRESHOLD=3        # Require 3 consecutive idle CPU checks
MEM_HIGH_COUNT_THRESHOLD=3    # Require 3 consecutive high memory checks
LOGFILE="soffice_monitor.log"

echo "Script started at $(date)" >> "$LOGFILE"

# Function: Measure instantaneous CPU usage for all soffice.bin processes.
# It calculates the delta in CPU ticks over a 1-second interval,
# then converts that into a percentage (similar to what 'top' shows).
get_instant_cpu_usage() {
    local clk_tck sum_before=0 sum_after=0 pid pids usage delta
    clk_tck=$(getconf CLK_TCK)
    pids=$(pgrep soffice.bin)
    if [ -z "$pids" ]; then
        echo "0"
        return
    fi

    # Sum CPU times (user + system) from /proc/<pid>/stat (fields 14 and 15)
    for pid in $pids; do
        if [ -f "/proc/$pid/stat" ]; then
            local proc_time
            proc_time=$(awk '{print $14+$15}' /proc/"$pid"/stat)
            sum_before=$(echo "$sum_before + $proc_time" | bc)
        fi
    done

    sleep 1

    for pid in $pids; do
        if [ -f "/proc/$pid/stat" ]; then
            local proc_time
            proc_time=$(awk '{print $14+$15}' /proc/"$pid"/stat)
            sum_after=$(echo "$sum_after + $proc_time" | bc)
        fi
    done

    delta=$(echo "$sum_after - $sum_before" | bc)
    # Instantaneous CPU usage (%) = (delta ticks / clk_tck) * 100
    usage=$(echo "scale=2; ($delta / $clk_tck) * 100" | bc -l)
    echo "$usage"
}

# Main monitoring loop
IDLE_COUNT=0
MEM_HIGH_COUNT=0

while true; do
    # Get list of soffice.bin PIDs
    PIDS=$(pgrep soffice.bin)
    if [ -z "$PIDS" ]; then
        echo "$(date) - No soffice.bin processes found" >> "$LOGFILE"
        IDLE_COUNT=0
        MEM_HIGH_COUNT=0
        sleep $CHECK_INTERVAL
        continue
    fi

    # Get total memory usage (%) for all soffice.bin processes
    MEM_USAGE=$(ps -C soffice.bin -o %mem --no-headers | awk '{sum+=$1} END {print sum}')
    if [ -z "$MEM_USAGE" ] || [ "$MEM_USAGE" = "0" ]; then
        echo "$(date) - No soffice.bin memory data found" >> "$LOGFILE"
        IDLE_COUNT=0
        MEM_HIGH_COUNT=0
        sleep $CHECK_INTERVAL
        continue
    fi

    # Get instantaneous CPU usage (via our custom function)
    CPU_USAGE=$(get_instant_cpu_usage)
    MEM_USAGE=$(echo "$MEM_USAGE" | xargs)
    CPU_USAGE=$(echo "$CPU_USAGE" | xargs)

    # Check if memory is too high and CPU usage is low
    mem_high=$(echo "$MEM_USAGE > $MAX_MEM_PERCENT" | bc -l)
    cpu_idle=$(echo "$CPU_USAGE < $CPU_USAGE_THRESHOLD" | bc -l)

    echo "$(date) - Total soffice.bin memory: ${MEM_USAGE}% | Instantaneous CPU: ${CPU_USAGE}% | MEM high count: $MEM_HIGH_COUNT | Idle CPU count: $IDLE_COUNT" >> "$LOGFILE"

    # Increment or reset counters based on current conditions
    if [ "$mem_high" -eq 1 ]; then
        MEM_HIGH_COUNT=$((MEM_HIGH_COUNT + 1))
    else
        MEM_HIGH_COUNT=0
    fi

    if [ "$cpu_idle" -eq 1 ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
    else
        IDLE_COUNT=0
    fi

    # If both high memory and low CPU persist for the required checks, perform final verification.
    if [ "$MEM_HIGH_COUNT" -ge "$MEM_HIGH_COUNT_THRESHOLD" ] && [ "$IDLE_COUNT" -ge "$IDLE_COUNT_THRESHOLD" ]; then
        echo "$(date) - Potential idle soffice.bin detected, performing final verification..." >> "$LOGFILE"
        ACTIVITY_DETECTED=0
        
        # Take 3 quick verification samples (1 second apart)
        for (( i=1; i<=3; i++ )); do
            sleep 1
            VERIFY_CPU=$(get_instant_cpu_usage)
            echo "$(date) - Verification sample $i Instantaneous CPU: ${VERIFY_CPU}%" >> "$LOGFILE"
            if (( $(echo "$VERIFY_CPU >= $CPU_USAGE_THRESHOLD" | bc -l) )); then
                echo "$(date) - Detected CPU activity (${VERIFY_CPU}%) during verification, aborting kill" >> "$LOGFILE"
                ACTIVITY_DETECTED=1
                break
            fi
        done

        if [ "$ACTIVITY_DETECTED" -eq 1 ]; then
            IDLE_COUNT=0
            MEM_HIGH_COUNT=0
            sleep $CHECK_INTERVAL
            continue
        fi

        # Final verification with a slightly longer pause
        sleep 2
        FINAL_CPU=$(get_instant_cpu_usage)
        echo "$(date) - Final verification Instantaneous CPU after 2-second pause: ${FINAL_CPU}%" >> "$LOGFILE"
        if (( $(echo "$FINAL_CPU >= $CPU_USAGE_THRESHOLD" | bc -l) )); then
            echo "$(date) - Detected CPU activity (${FINAL_CPU}%) during final verification, aborting kill" >> "$LOGFILE"
            IDLE_COUNT=0
            MEM_HIGH_COUNT=0
        else
            echo "$(date) - Killing idle soffice.bin processes (PIDs: $PIDS) (Memory: ${MEM_USAGE}%, Instantaneous CPU: ${CPU_USAGE}%, MEM high for ${MEM_HIGH_COUNT} checks, Idle CPU for ${IDLE_COUNT} checks)" >> "$LOGFILE"
            for pid in $PIDS; do
                # Capture the output of the kill command
                kill_output=$(sudo kill -9 "$pid" 2>&1)
                if [ -n "$kill_output" ]; then
                    echo "$(date) - Error killing pid $pid: $kill_output" >> "$LOGFILE"
                else
                    echo "$(date) - Successfully killed pid $pid" >> "$LOGFILE"
                fi
            done
            IDLE_COUNT=0
            MEM_HIGH_COUNT=0
            sleep 10  # Pause briefly after killing
        fi
    fi

    # Keep the log file trimmed to the last 200 lines
    if [ -f "$LOGFILE" ]; then
        LINE_COUNT=$(wc -l < "$LOGFILE")
        if [ "$LINE_COUNT" -gt 200 ]; then
            tail -n 200 "$LOGFILE" > "${LOGFILE}.tmp" && mv "${LOGFILE}.tmp" "$LOGFILE"
        fi
    fi

    sleep $CHECK_INTERVAL
done
