"""Optional local web UI: job monitoring, preview, and a timeline editor.

Runs as its own process against the same workspace and job store the MCP server
uses, so both can run at once, or the UI can run standalone with no MCP client
attached.
"""
