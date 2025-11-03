#!/bin/bash
# Send start/end budget summary via email

set -euo pipefail

TMP_ACCINFO_AFTER="${ACCINFO_DIR}/budget_after_$(date +%Y%m%d_%H%M%S).txt"
TMP_EMAIL="${ACCINFO_DIR}/budget_report_$(date +%Y%m%d_%H%M%S).txt"

extract_budget_summary() {
    # Extract only the three lines of interest
    awk '
        /Budget left for dispatch/ { print; next }
        /Budget left for submit/ { print; next }
        /^\(Checked at/ { print; next }
    ' "$1"
}

echo "Collecting final budget overview for $PARTITION_RUN..."

# Capture the final snapshot
if command -v budget-overview &>/dev/null; then
    budget-overview -p "$PARTITION_RUN" > "$TMP_ACCINFO_AFTER"
else
    echo "Warning: budget-overview not found; using accinfo fallback."
    accinfo > "$TMP_ACCINFO_AFTER"
fi

# Create the email report
{
    echo "===== Budget overview at START ====="
    extract_budget_summary "$TMP_ACCINFO_BEFORE"
    echo ""
    echo "===== Budget overview at END ====="
    extract_budget_summary "$TMP_ACCINFO_AFTER"
    echo ""
    echo "Host: $(hostname)"
    echo "User: $(whoami)"
    echo "Date: $(date)"
} > "$TMP_EMAIL"

# Send via mail or print fallback
if command -v mail &>/dev/null; then
    mail -s "[LCDB] Budget summary report ($PARTITION_RUN)" "$EMAILS" < "$TMP_EMAIL"
    echo "Budget summary email sent to $EMAILS"
else
    echo "mail command not found; printing email contents below:"
    cat "$TMP_EMAIL"
fi
