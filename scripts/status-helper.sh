#!/bin/bash

# Status Helper Script
# Provides auto-cleanup and management functions for .claude-current-status

STATUS_FILE=".claude-current-status"
AUTO_CONDENSE_LIMIT=300  # Auto-condense when file exceeds this many lines

# Function: Check if status file needs condensation
check_and_condense_if_needed() {
    if [ ! -f "$STATUS_FILE" ]; then
        return 0
    fi

    CURRENT_LINES=$(wc -l < "$STATUS_FILE")
    if [ "$CURRENT_LINES" -gt "$AUTO_CONDENSE_LIMIT" ]; then
        echo "⚠️  Status file large ($CURRENT_LINES lines > $AUTO_CONDENSE_LIMIT limit)"
        echo "   Consider running: /condense 200"
        return 1
    fi
    return 0
}

# Function: Add entry to status file with auto-cleanup check
add_status_entry() {
    local entry="$1"
    local section="${2:-Update}"

    # Check if file needs condensation
    check_and_condense_if_needed

    # Add timestamp and entry
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "" >> "$STATUS_FILE"
    echo "## $section ($timestamp)" >> "$STATUS_FILE"
    echo "$entry" >> "$STATUS_FILE"

    echo "✅ Added to status file: $section"
}

# Function: Quick status summary
status_summary() {
    if [ ! -f "$STATUS_FILE" ]; then
        echo "❌ Status file not found"
        return 1
    fi

    local total_lines=$(wc -l < "$STATUS_FILE")
    local mod_time=$(stat -c "%y" "$STATUS_FILE" 2>/dev/null || stat -f "%Sm" "$STATUS_FILE" 2>/dev/null)

    echo "📊 Status File Summary:"
    echo "   File: $STATUS_FILE"
    echo "   Lines: $total_lines"
    echo "   Modified: $mod_time"
    echo "   Auto-condense limit: $AUTO_CONDENSE_LIMIT lines"

    if [ "$total_lines" -gt "$AUTO_CONDENSE_LIMIT" ]; then
        echo "   ⚠️  Exceeds limit by $((total_lines - AUTO_CONDENSE_LIMIT)) lines"
    else
        echo "   ✅ Within limit ($((AUTO_CONDENSE_LIMIT - total_lines)) lines remaining)"
    fi
}

# Function: Show recent entries
show_recent() {
    local lines="${1:-20}"
    if [ ! -f "$STATUS_FILE" ]; then
        echo "❌ Status file not found"
        return 1
    fi

    echo "📄 Recent entries (last $lines lines):"
    echo "======================================"
    tail -n "$lines" "$STATUS_FILE"
}

# Main execution
case "${1:-}" in
    "check")
        check_and_condense_if_needed
        ;;
    "add")
        if [ -z "$2" ]; then
            echo "Usage: $0 add \"entry text\" [section]"
            exit 1
        fi
        add_status_entry "$2" "${3:-Update}"
        ;;
    "summary")
        status_summary
        ;;
    "recent")
        show_recent "${2:-20}"
        ;;
    *)
        echo "Status Helper - Manage .claude-current-status"
        echo "Usage:"
        echo "  $0 check              - Check if file needs condensation"
        echo "  $0 add \"text\" [section] - Add entry with optional section"
        echo "  $0 summary            - Show file statistics"
        echo "  $0 recent [lines]     - Show recent entries (default: 20)"
        echo ""
        echo "Auto-condense limit: $AUTO_CONDENSE_LIMIT lines"
        ;;
esac