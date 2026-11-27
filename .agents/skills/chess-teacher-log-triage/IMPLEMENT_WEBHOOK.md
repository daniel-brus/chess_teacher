# Implement Automation via webhook (issue labels)

Cursor’s Automations UI often cannot attach a true **issue** label trigger. Workaround:

1. Implement Automation trigger = **Webhook**
   URL example: `https://api2.cursor.sh/automations/webhook/<id>`
2. In Automations UI → webhook → **Generate auth header** → copy the Bearer token.
3. GitHub repo → Settings → Secrets and variables → Actions → add:
   - `CURSOR_IMPLEMENT_WEBHOOK_URL` = the webhook URL
   - `CURSOR_IMPLEMENT_WEBHOOK_SECRET` = the Bearer token only (no `Bearer ` prefix)
4. Workflow [`.github/workflows/cursor-implement-webhook.yml`](../../../.github/workflows/cursor-implement-webhook.yml) runs on `issues: labeled` when the label is `agent-fix` and POSTs JSON `{ "context": "..." }`.

## Manual test

```bash
curl -X POST "$CURSOR_IMPLEMENT_WEBHOOK_URL" \
  -H "Authorization: Bearer $CURSOR_IMPLEMENT_WEBHOOK_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"context":"Implement GitHub issue #129. Follow AUTOMATION_IMPLEMENT.md."}'
```

## Security

Treat the webhook URL + secret like credentials. Rotate the auth header in Cursor if leaked.
