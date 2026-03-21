# Searchat - CLAUDE.md Template

Add this section to your global `~/.claude/CLAUDE.md` to enable conversation history search.

---

## Conversation History Search

Search past Claude Code conversations via local API (requires server running).

**Search:**
```bash
curl -s "http://localhost:8000/api/search?q=QUERY&limit=5" | jq '.results[] | {id: .conversation_id, title, snippet}'
```

**Get full conversation:**
```bash
curl -s "http://localhost:8000/api/conversation/CONVERSATION_ID" | jq '.messages[] | {role, content: .content[:500]}'
```

**When to use:**
- User asks "did we discuss X before" or "find that conversation about Y"
- Looking for previous solutions to similar problems
- Checking how something was implemented in past sessions

**Parameters:**
- `q` — search query (supports natural language)
- `mode` - cross-layer (default), distill, verbatim
- `limit` — max results (1-100, default 100)
- `project` — filter by project name

**Start server:** `searchat-web`


