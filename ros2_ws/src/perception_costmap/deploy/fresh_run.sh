#!/usr/bin/env bash
# fresh_run.sh -- drop all accumulated state between runs, without restarting.
#
# WHY THIS EXISTS
# For IGVC each run must start carrying nothing from the last one. Almost
# everything in this stack is already ephemeral by construction:
#
#   - nothing is written to disk: no map_server, no map_saver, no SLAM
#   - both Nav2 costmaps are rolling_window with NO static layer
#   - STVL is mapping_mode:false and decays observations in ~3 s
#
# The one exception is perception's per-cell temporal confidence, which
# accumulates across ticks by design and lives as long as the process. Until
# now the only way to clear it was restarting the whole stack, which happened
# to work but is not a procedure you want between competition runs.
#
# Run this after each run, and before the first one. It takes about a second.
#
#   ./fresh_run.sh              # reset perception + both Nav2 costmaps
#   ./fresh_run.sh --perception # perception only (Nav2 not running yet)
#
# Env: inherits ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI from the
# shell. On the car those must already be set -- see FINALIZE.md's standing
# rules -- and this script deliberately does not override them, so it resets
# whatever stack your shell is actually talking to.
set -uo pipefail

PERCEPTION_ONLY=0
[ "${1:-}" = "--perception" ] && PERCEPTION_ONLY=1

TIMEOUT="${FRESH_RUN_TIMEOUT:-10}"
failed=0

have_service() {
    ros2 service list 2>/dev/null | grep -qx "$1"
}

call() {  # name type request label critical
    local name="$1" type="$2" req="$3" label="$4" critical="$5"
    if ! have_service "$name"; then
        if [ "$critical" = "1" ]; then
            printf '  [MISSING ] %-28s %s is not advertised\n' "$label" "$name"
            failed=1
        else
            printf '  [skipped ] %-28s not running\n' "$label"
        fi
        return
    fi
    local out
    if out=$(timeout "$TIMEOUT" ros2 service call "$name" "$type" "$req" 2>&1); then
        # Trigger replies carry success=False on a refusal, which is not a
        # non-zero exit -- check the payload, not just the exit code.
        if grep -q 'success=False' <<<"$out"; then
            printf '  [REFUSED ] %-28s %s\n' "$label" \
                "$(grep -o "message='[^']*'" <<<"$out" | head -1)"
            failed=1
        else
            printf '  [ ok     ] %-28s\n' "$label"
            grep -o "message='[^']*'" <<<"$out" | head -1 | sed "s/^/               /"
        fi
    else
        printf '  [FAILED  ] %-28s call timed out or errored\n' "$label"
        failed=1
    fi
}

echo "=============================================================="
echo " FRESH RUN -- clearing accumulated state"
echo " domain=${ROS_DOMAIN_ID:-0}  rmw=${RMW_IMPLEMENTATION:-default}"
echo "=============================================================="

call /perception/reset std_srvs/srv/Trigger '{}' "perception temporal filters" 1

if [ "$PERCEPTION_ONLY" = "0" ]; then
    call /local_costmap/clear_entirely_local_costmap \
         nav2_msgs/srv/ClearEntireCostmap '{}' "nav2 local costmap" 0
    call /global_costmap/clear_entirely_global_costmap \
         nav2_msgs/srv/ClearEntireCostmap '{}' "nav2 global costmap" 0
fi

echo "--------------------------------------------------------------"
if [ "$failed" = "0" ]; then
    echo " CLEAR -- no prior-run state retained."
    echo " STVL (lidar voxels) is not cleared here: mapping_mode:false with"
    echo " voxel_decay 3.0 s means it ages out on its own within ~3 seconds."
else
    echo " NOT CLEAR -- something above failed. Do NOT start a scored run."
    echo " Fall back to a full stack restart (deploy/full_stack_restart.sh)."
fi
echo "=============================================================="
exit "$failed"
