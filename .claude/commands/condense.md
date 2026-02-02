---
allowed-tools: [Bash]
description: Condense .claude-current-status by archiving old content and keeping recent work
approach: script-delegation
token-cost: ~150 (archives old content, keeps recent work)
best-for: Managing file size while preserving recent context
---

# Condense - Manage Status File Size

Archives older content from `.claude-current-status` to keep the file manageable while preserving recent work context.

## How It Works

1. Checks current file size and shows statistics
2. Archives content beyond the keep limit (default: 200 lines) to `.claude-archive/`
3. Creates new condensed file with recent content only
4. Preserves file structure and session headings

## Your Task

Run the condense script to manage file size:

```bash
#!/bin/bash

STATUS_FILE=".claude-current-status"
ARCHIVE_DIR=".claude-archive"
KEEP_LINES="${1:-200}"  # Default: keep last 200 lines

echo "📄 Claude Current Status - File Condensation"
echo "============================================"
echo ""

if [ ! -f "$STATUS_FILE" ]; then
    echo "❌ Status file not found: $STATUS_FILE"
    echo "   No action needed."
    exit 0
fi

# Get file statistics
TOTAL_LINES=$(wc -l < "$STATUS_FILE")
MOD_TIME=$(stat -c "%y" "$STATUS_FILE" 2>/dev/null || stat -f "%Sm" "$STATUS_FILE" 2>/dev/null)

echo "📊 Current Status:"
echo "   File: $STATUS_FILE"
echo "   Total lines: $TOTAL_LINES"
echo "   Modified: $MOD_TIME"
echo "   Keep limit: $KEEP_LINES lines"
echo ""

if [ "$TOTAL_LINES" -le "$KEEP_LINES" ]; then
    echo "✅ File size ($TOTAL_LINES lines) is within keep limit ($KEEP_LINES lines)."
    echo "   No condensation needed."
    exit 0
fi

LINES_TO_ARCHIVE=$((TOTAL_LINES - KEEP_LINES))
ARCHIVE_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
ARCHIVE_FILE="$ARCHIVE_DIR/status-archive-$ARCHIVE_TIMESTAMP.txt"

echo "📦 File exceeds limit by $LINES_TO_ARCHIVE lines."
echo "   Preparing to archive older content..."
echo ""

# Create archive directory if needed
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "📁 Creating archive directory: $ARCHIVE_DIR"
    mkdir -p "$ARCHIVE_DIR"
fi

# Extract and archive older content (lines 1 through LINES_TO_ARCHIVE)
echo "🗂️  Archiving lines 1-$LINES_TO_ARCHIVE to: $ARCHIVE_FILE"
head -n "$LINES_TO_ARCHIVE" "$STATUS_FILE" > "$ARCHIVE_FILE"

# Create temporary file with recent content
TEMP_FILE=$(mktemp)
echo "📝 Keeping lines $((LINES_TO_ARCHIVE + 1))-$TOTAL_LINES (recent $KEEP_LINES lines)"
tail -n "$KEEP_LINES" "$STATUS_FILE" > "$TEMP_FILE"

# Add archive note at top of condensed file
echo "🔍 Adding archive reference to condensed file..."
{
    echo "# Claude Current Status - Condensed"
    echo ""
    echo "## Archive Note"
    echo "- **Condensed on**: $(date)"
    echo "- **Original size**: $TOTAL_LINES lines"
    echo "- **Archived**: $LINES_TO_ARCHIVE lines to $ARCHIVE_FILE"
    echo "- **Kept**: $KEEP_LINES lines (recent work)"
    echo "- **Previous archive**: Lines 1-$LINES_TO_ARCHIVE"
    echo ""
    echo "## Recent Work (Last $KEEP_LINES lines)"
    echo "========================================"
    echo ""
    cat "$TEMP_FILE"
} > "$STATUS_FILE"

# Clean up
rm "$TEMP_FILE"

# Report results
NEW_LINES=$(wc -l < "$STATUS_FILE")
echo ""
echo "✅ Condensation complete:"
echo "   - Archived: $LINES_TO_ARCHIVE lines to $ARCHIVE_FILE"
echo "   - Kept: $KEEP_LINES lines in main file"
echo "   - New file size: $NEW_LINES lines"
echo "   - Reduction: $((TOTAL_LINES - NEW_LINES)) lines"
echo ""
echo "📁 Archive directory contents:"
ls -la "$ARCHIVE_DIR/" 2>/dev/null || echo "   (empty)"
echo ""
echo "💡 Use '/continue' to view recent context. Run '/condense N' to keep N lines."
```

## Notes

- **Safe operation**: Creates archives before modifying original file
- **Configurable**: Use `/condense 300` to keep 300 lines instead of default 200
- **Preserves context**: Recent work remains accessible via `/continue`
- **Archive tracking**: Each condensation creates timestamped archive file
- **Reversible**: Archived content preserved in `.claude-archive/`

## Auto-Cleanup Integration

For automatic cleanup during updates, add this check before modifying `.claude-current-status`:

```bash
# In your update scripts, check file size and optionally condense
if [ $(wc -l < .claude-current-status) -gt 400 ]; then
    echo "⚠️ Status file large, consider running: /condense"
    # Or auto-condense: /condense 250
fi
```

## Usage

```bash
/condense          # Keep last 200 lines (default)
/condense 300      # Keep last 300 lines
/condense 150      # Keep last 150 lines
```

Run when `.claude-current-status` gets large (>300 lines) to maintain performance.