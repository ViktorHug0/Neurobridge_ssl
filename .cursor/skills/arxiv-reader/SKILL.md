---
name: arxiv-reader
description: Fetch and read arXiv papers as Markdown using arxiv2md.org. Use when working with arXiv papers, paper URLs, or when requested to summarize or analyze arXiv content.
---

# arXiv Reader

Fetch and read arXiv papers efficiently by converting their URLs to Markdown format via `arxiv2md.org`.

## Instructions

When you encounter an arXiv URL or need to read an arXiv paper:

1. **Identify the arXiv ID**: Extract the ID from the URL (e.g., `2501.11120v1`).
2. **Transform the URL**: Add `2md` after `arxiv` in the hostname.
   - Original: `https://arxiv.org/abs/2501.11120v1`
   - Modified: `https://arxiv2md.org/abs/2501.11120v1`
3. **Fetch the Content**: Use the `WebFetch` tool with the modified URL to retrieve the paper in Markdown format.
4. **Analyze the Paper**: Once fetched, you can summarize, extract key findings, or answer specific questions about the paper's content.

## Examples

### URL Conversion
- `https://arxiv.org/abs/2303.17564` → `https://arxiv2md.org/abs/2303.17564`
- `https://arxiv.org/pdf/2401.00001.pdf` → `https://arxiv2md.org/abs/2401.00001` (Note: target the `/abs/` route for the best Markdown conversion)

### Usage Workflow
1. User: "Summarize this paper: https://arxiv.org/abs/2501.11120v1"
2. Agent: Transforms to `https://arxiv2md.org/abs/2501.11120v1`
3. Agent: Calls `WebFetch(url="https://arxiv2md.org/abs/2501.11120v1")`
4. Agent: Summarizes the returned Markdown content.
