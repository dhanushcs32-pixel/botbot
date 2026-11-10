# Prompt for Cline — CoT diagnostic mode in the Bot Bot app

## Context — read before starting

The backend already has a working CoT diagnostic endpoint. You are adding
the app-side feature that consumes it. Do NOT touch the backend in this
task.

**Backend contract (verified live, from the addon implementation):**

- `POST {BASE_URL}/chat/cot`
- Required headers: `X-API-Key: <key>` AND `X-CoT-Activate: diagnostic-replay`
  (exact value; missing/wrong header → 400). The endpoint is PRIMARY-KEY-ONLY —
  the guest key gets 403.
- Body: `{"message": "...", "session_id": "..."}`
- Success response (REST, single response, NOT streaming/websocket):
  `{"mode": "...", "reasoning": {"content": "...", "note": "model's self-reported reasoning — NOT verified to reflect its actual internal process; diagnostic signal only"}, "reply": {"content": "..."}, "measured_tokens": ..., "duration_s": ..., "num_predict": 700, "parse": "ok"}`
- Parse failure response: `{"error": "cot_parse_failed", "detail": "...", "raw_output": "...", "note": ...}`
- This call can take SEVERAL MINUTES (backend HTTP timeout is scaled to 900s
  for this endpoint specifically; NPU generations run ~1 token/s). Slow is
  expected and acceptable.
- The backend does NOT persist CoT turns (no save_turn) — they exist only in
  the response.
- **CORS is already handled backend-side and verified live:** preflight for
  `capacitor://localhost` and `https://localhost` against /chat/cot returns
  200 with the correct allow-headers (x-api-key, x-cot-activate). You do not
  need any backend change for CORS.

**Reference implementation:** `C:\botbot\chat-cot.html` on the backend machine —
a working desktop client for this exact endpoint. Mirror its request shape,
headers, and response handling. Don't reinvent.

**Before coding, re-read the app audit report** (websocket/ack/dedupe/
offline-queue/reconnect were confirmed sound; Fix A = E2EE copy, Fix B =
typing indicator). Every one of those areas is off-limits in this task.

## HARD CONSTRAINTS

1. The normal chat pipeline (websocket connection, chat_message flow,
   ack/dedupe, offline queue, reconnect) must NOT be modified or routed
   through. CoT mode is a separate one-shot REST call path.
2. Normal chat with the toggle OFF must behave exactly as it did after the
   Fix A/B pass — confirmed in verification, not assumed.
3. Fix A (E2EE copy) and Fix B (typing indicator) changes must not be
   reverted or altered. Fix B's typing indicator is tied to the websocket
   path — do NOT reuse it for CoT (different mechanism, different call
   shape).
4. Do not add new session-management logic — CoT uses whatever session_id
   is currently active.
5. No Phase 2 agentic RAG work, no backend changes.

## Step 0 — Confirm the HTTP transport (do first, brief report)

Check how the app currently makes authenticated REST calls with the
X-API-Key header (e.g. /history): Capacitor Http plugin (native,
CORS-exempt) or window.fetch from the WebView? Use the SAME transport for
CoT. The CORS preflight for the WebView fetch path is already permitted
backend-side (verified above), so either transport works — just be
consistent with the app's existing pattern.

## Step 1 — Explicit toggle, OFF by default

- Add "CoT Diagnostic Mode (primary key only)" to the existing Settings
  screen, alongside base URL / API key fields. Switch control, OFF default.
- Persist it the same way base URL/session id are persisted today (match
  the audit's finding — localStorage or equivalent — same key-naming
  convention).
- Description text under the toggle, visible in the UI: "Sends messages to
  a special diagnostic endpoint that also returns the model's self-narrated
  reasoning before the reply. Takes several minutes per message. Requires
  the primary API key (guest keys are rejected server-side). The reasoning
  is a self-report, NOT verified to reflect the model's actual process."

## Step 2 — Separate send path when ON

- Toggle OFF → send button behaves exactly as today (websocket
  chat_message flow, untouched code path).
- Toggle ON → send button POSTs to `{BASE_URL}/chat/cot` with the two
  headers and body `{"message": ..., "session_id": <currently active
  session id>}`. No websocket send, no ack/dedupe, no offline queue
  involvement. If the websocket happens to be connected, leave it
  connected and untouched — CoT simply doesn't use it.
- While the toggle is ON, show a persistent banner/indicator in the chat
  view (e.g. amber strip: "CoT diagnostic mode — reasoning is a self-
  report; replies may take several minutes") so the mode is impossible to
  forget is on. It must disappear when the toggle is turned off.

## Step 3 — Rendering: reasoning bubble first, then reply

- New bubble type for reasoning, visually distinct from normal bot replies
  (e.g. amber/darker variant with a left border, mirroring the desktop
  chat-cot.html client), content prefixed in the UI with something like
  "🧠 reasoning — self-report, not verified" AND the note from the
  response's reasoning.note field displayed near the content.
- Reply renders after it as a normal bot bubble.
- On `cot_parse_failed`: render a distinct error state showing the failure
  detail and the full `raw_output` in an inspectable way (expandable or
  scrollable block) — never styled as a normal reply, never silently
  swallowed.
- CoT reasoning/reply are ephemeral, render-only entries in the transcript
  view: do NOT add them to the offline queue or any persisted message
  store (the backend doesn't save them either). On reload they simply
  won't be there — that's correct.

## Step 4 — Long-request UX

- Distinct loading state for the CoT call ("CoT generating — may take
  several minutes…"), NOT Fix B's websocket typing indicator.
- Client-side timeout ≥ 950s (the backend allows up to 900s of generation).
- Network failure / timeout → visible error state. No hang, no crash.
- 401/403 responses → visible, specific errors ("requires the primary API
  key" for 403 on this endpoint), never silently swallowed.

## Verify (all of it, with evidence)

1. Toggle OFF: normal send/receive, ack/dedupe, offline queue behave
   exactly as after the Fix A/B pass — re-run the Fix A/B manual
   verification steps and confirm.
2. Toggle ON: send a real message; confirm (via response inspection and/or
   the backend's own log showing the long /api/chat generation) that it
   hit /chat/cot; confirm the reasoning bubble renders first, then the
   reply bubble, with the non-authoritative note visible.
3. Guest/limited key: configure a non-primary key, attempt a CoT send,
   confirm the 403 surfaces as a visible error explaining the primary-key
   requirement.
4. Parse-failure: force it client-side — e.g. temporarily intercept the
   response in a debug build (or point at a local mock returning
   `{"error": "cot_parse_failed", "detail": "missing delimiter(s)",
   "raw_output": "..."}`) and confirm the error state renders with
   raw_output inspectable. Report the method you used.
5. `npm run lint`, `npm run build`, `npx cap sync android`, `assembleDebug`
   — all clean.

## Report back

Files changed; toggle location, behavior, and the exact persistence key
used; evidence that normal chat is unaffected (real send/receive + Fix A/B
checks); the real end-to-end CoT result (reasoning + reply as displayed,
duration); 403 and parse-failure handling evidence. Same evidentiary
standard as the app audit — real outputs, not "should work."

