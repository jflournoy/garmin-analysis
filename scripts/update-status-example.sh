#!/bin/bash

# Example: How to update .claude-current-status with auto-cleanup check

STATUS_FILE=".claude-current-status"
AUTO_CONDENSE_LIMIT=300

# Function to add entry with auto-cleanup check
add_status_with_cleanup_check() {
    local entry="$1"
    local section="${2:-Update}"

    # Check file size before adding
    if [ -f "$STATUS_FILE" ]; then
        CURRENT_LINES=$(wc -l < "$STATUS_FILE")
        if [ "$CURRENT_LINES" -gt "$AUTO_CONDENSE_LIMIT" ]; then
            echo "⚠️  Status file large ($CURRENT_LINES lines). Consider:"
            echo "   /condense 200  # Keep last 200 lines"
            echo "   /condense 250  # Keep last 250 lines"
            echo ""
            echo "Proceeding with update anyway..."
        fi
    fi

    # Add the entry
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "" >> "$STATUS_FILE"
    echo "## $section ($timestamp)" >> "$STATUS_FILE"
    echo "$entry" >> "$STATUS_FILE"

    echo "✅ Added: $section"
}

# Example usage
echo "Example: Updating status with auto-cleanup check"
echo "================================================="
echo ""

# Check current status
if [ -f "$STATUS_FILE" ]; then
    echo "📊 Current file: $STATUS_FILE"
    echo "   Lines: $(wc -l < "$STATUS_FILE")"
    echo "   Auto-condense limit: $AUTO_CONDENSE_LIMIT lines"
    echo ""
fi

# Add an example entry
add_status_with_cleanup_check "This is an example entry showing the auto-cleanup pattern." "Example"

echo ""
echo "💡 Best practices:"
echo "1. Run '/continue' to see recent context before starting work"
echo "2. Update .claude-current-status as you work"
echo "3. If file exceeds $AUTO_CONDENSE_LIMIT lines, run '/condense'"
echo "4. Use 'scripts/status-helper.sh' for programmatic updates"