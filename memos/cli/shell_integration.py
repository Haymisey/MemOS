"""Shell integration logic for MemOS.

Generates initialization scripts for various shells (Bash, Zsh, PowerShell)
to automatically capture command failures and ingest them into MemOS.
"""

from __future__ import annotations

import textwrap

def generate_bash_hook(api_url: str = "http://localhost:11437") -> str:
    """Generate a Bash shell hook using PROMPT_COMMAND."""
    return textwrap.dedent(f"""
        # --- MemOS Bash Hook ---
        _memos_hook() {{
            local exit_code=$?
            local last_cmd=$(history 1 | sed 's/^[ ]*[0-9]*[ ]*//')
            
            if [ $exit_code -ne 0 ]; then
                # Check if command is not 'memos' itself to avoid loops
                if [[ ! "$last_cmd" =~ ^memos ]]; then
                    (curl -s -X POST "{api_url}/v1/memories" \\
                        -H "Content-Type: application/json" \\
                        -d "{{\\"content\\": \\"Command failed with exit code $exit_code: $last_cmd\\", \\"source\\": \\"terminal_error\\", \\"tags\\": [\\"cli\\", \\"error\\"]}}" \\
                        >/dev/null 2>&1 &)
                fi
            fi
        }}
        if [[ ! "$PROMPT_COMMAND" =~ _memos_hook ]]; then
            PROMPT_COMMAND="_memos_hook;$PROMPT_COMMAND"
        fi
    """).strip()

def generate_zsh_hook(api_url: str = "http://localhost:11437") -> str:
    """Generate a Zsh shell hook using add-zsh-hook."""
    return textwrap.dedent(f"""
        # --- MemOS Zsh Hook ---
        autoload -Uz add-zsh-hook

        _memos_preexec() {{
            _MEMOS_LAST_CMD="$1"
        }}

        _memos_precmd() {{
            local exit_code=$?
            if [ $exit_code -ne 0 ] && [ -n "$_MEMOS_LAST_CMD" ]; then
                # Avoid capturing 'memos' commands
                if [[ ! "$_MEMOS_LAST_CMD" =~ ^memos ]]; then
                    (curl -s -X POST "{api_url}/v1/memories" \\
                        -H "Content-Type: application/json" \\
                        -d "{{\\"content\\": \\"Command failed with exit code $exit_code: $_MEMOS_LAST_CMD\\", \\"source\\": \\"terminal_error\\", \\"tags\\": [\\"cli\\", \\"error\\"]}}" \\
                        >/dev/null 2>&1 &)
                fi
            fi
            unset _MEMOS_LAST_CMD
        }}

        add-zsh-hook preexec _memos_preexec
        add-zsh-hook precmd _memos_precmd
    """).strip()

def generate_powershell_hook(api_url: str = "http://localhost:11437") -> str:
    """Generate a PowerShell hook using the prompt function."""
    return textwrap.dedent(f"""
        # --- MemOS PowerShell Hook ---
        function _memos_prompt {{
            $last_exit_code = $LASTEXITCODE
            if ($null -ne $last_exit_code -and $last_exit_code -ne 0) {{
                $history = Get-History -Count 1
                if ($null -ne $history) {{
                    $last_cmd = $history.CommandLine
                    # Avoid capturing 'memos' commands
                    if ($last_cmd -notmatch "^memos") {{
                        $body = @{{
                            content = "Command failed with exit code $last_exit_code: $last_cmd"
                            source = "terminal_error"
                            tags = @("cli", "error")
                        }} | ConvertTo-Json
                        
                        Start-Job -ScriptBlock {{
                            param($url, $json)
                            Invoke-RestMethod -Uri "$url/v1/memories" -Method Post -Body $json -ContentType "application/json" | Out-Null
                        }} -ArgumentList "{api_url}", $body | Out-Null
                    }}
                }}
            }}
            # Return standard prompt
            "PS $($ExecutionContext.SessionState.Path.CurrentLocation)> "
        }}

        # Backup old prompt if not already done
        if (!(Test-Path Function:\\_memos_old_prompt)) {{
            if (Test-Path Function:\\prompt) {{
                Rename-Item Function:\\prompt _memos_old_prompt
            }} else {{
                function _memos_old_prompt {{ "PS > " }}
            }}
        }}

        function prompt {{
            _memos_prompt
            _memos_old_prompt
        }}
    """).strip()
