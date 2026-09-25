---
name: awesome-tool-finder
description: Recommends AI coding tools by need and category using the bundled awesome-ai-coding-tools curated list. Use when the user asks for recommendations of AI coding tools, editors, coding agents, CLI assistants, code review/testing tools, app builders, or comparisons like "what should I use for X".
---

# AI Tool Finder

Help the user pick AI coding tools from the curated awesome-ai-coding-tools list.

## Steps

1. Read `references/awesome-ai-coding-tools.md` (bundled next to this file). It is a categorized list of ~18 sections: Code Editors and Assistants, Code Completion, Coding Agents, CLI Tools, App Builders, UI Generators, Code Review and Refactoring, Testing and QA, Code Search/Documentation/Code Models, Developer Productivity, AI Frameworks/SDKs & Local LLM Tools, and DevOps/Infrastructure.
2. Match the user's need to one or two categories. If their need is vague, ask one clarifying question (e.g. budget, open-source preference, editor/OS) before listing.
3. Recommend 3–5 tools max: name, one-line description from the list, and its link. Lead with the best fit and say why it fits their stated need. Note license/pricing when the list entry mentions it (open-source vs commercial).
4. If the bundled list seems stale for a fast-moving need (e.g. brand-new tools), you may optionally verify with a web search, but the bundled list is the primary source — do not require network access.

## Output format

A short intro sentence, then a compact list of recommendations with links, then one line of guidance on how to choose between the top two. Keep it under ~200 words unless the user asks for a full category survey.
