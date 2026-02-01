#!/bin/bash
if [[ ! -s "claude_instructions.txt" ]]; then
    echo "claude_instructions.txt doesn't exist or is empty"
    exit 1
fi
claude "follow the instructions in claude_instructions.txt"
