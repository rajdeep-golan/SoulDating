# Zoey voice backend

This is the voice agent behind the **"Call Zoey"** button on [zoeyai.love](https://www.zoeyai.love).

It is a [LiveKit Agents](https://docs.livekit.io/agents) worker. It connects to the LiveKit Cloud project and answers every call the website starts.

- **Speech-to-text:** Deepgram
- **AI model:** Groq (`openai/gpt-oss-120b`)
- **Voice:** ElevenLabs when a valid `sk_…` key is set, otherwise Deepgram Aura

When a call ends, the agent extracts the caller's profile from the transcript (`profile_capture.py`). It saves a copy under `logs/profiles/` and sends the profile to the website's database at `/api/profiles`.

> **"Call Zoey" only works while this backend is running.** It doesn't matter whether it runs on a Mac or on LiveKit Cloud. If no backend is running, the call screen opens but Zoey never joins.

---

## 1. One-time setup

Install Python 3.12 (newer versions can't build some audio packages):

```bash
brew install python@3.12
```

Create the virtual environment and install the dependencies:

```bash
cd ~/Zoey/SoulDating
/opt/homebrew/bin/python3.12 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python server.py download-files
```

Create a `.env` file in this folder. **Never commit it.** It needs these variables:

```dotenv
LIVEKIT_URL=wss://soul-ai-w0zf1p2s.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
ELEVEN_API_KEY=...            # optional; must start with sk_ or the Deepgram voice is used
ZOEY_INGEST_URL=https://www.zoeyai.love/api/profiles
ZOEY_INGEST_SECRET=...        # must match INGEST_SECRET in the Vercel project
```

## 2. Run it once (for testing)

```bash
cd ~/Zoey/SoulDating
venv/bin/python server.py start
```

It's ready when the log shows `registered worker`. Press `Ctrl+C` to stop it. Use `dev` instead of `start` for auto-reload while editing code.

## 3. Run it as an always-on service on a Mac (for presentations)

A macOS LaunchAgent keeps Zoey running in the background:

- It restarts Zoey automatically if it crashes, and starts it again when you log in.
- It keeps the Mac awake with `caffeinate` while Zoey is running.

The service file lives at `~/Library/LaunchAgents/love.zoeyai.agent.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>love.zoeyai.agent</string>
  <key>WorkingDirectory</key><string>/Users/karan/Zoey/SoulDating</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-dims</string>
    <string>/Users/karan/Zoey/SoulDating/venv/bin/python</string><string>server.py</string><string>start</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>/Users/karan/Zoey/SoulDating/logs/worker.log</string>
  <key>StandardErrorPath</key><string>/Users/karan/Zoey/SoulDating/logs/worker.log</string>
</dict></plist>
```

> The project must live **outside** `~/Documents`, `~/Desktop` and `~/Downloads`. macOS blocks background services from those folders, and the service fails with exit code 78.

**Start** the service:

```bash
launchctl enable gui/$(id -u)/love.zoeyai.agent
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/love.zoeyai.agent.plist
```

**Check that it's running.** You should see `state = running` and a `pid`:

```bash
launchctl print gui/$(id -u)/love.zoeyai.agent | grep -E 'state =|pid ='
```

**Watch the log.** `registered worker` means it's ready, and `received job request` means a call came in:

```bash
tail -f ~/Zoey/SoulDating/logs/worker.log
```

**Restart** it, for example after changing code or `.env`:

```bash
launchctl kickstart -k gui/$(id -u)/love.zoeyai.agent
```

**Stop** the service. This also stops it from starting at login:

```bash
launchctl bootout gui/$(id -u)/love.zoeyai.agent
```

```bash
launchctl disable gui/$(id -u)/love.zoeyai.agent
```

### Keeping the Mac from stopping it

- Keep the Mac **plugged in** with the **lid open**. Closing the lid sleeps a MacBook even with `caffeinate` running.
- Don't restart, log out or install macOS updates during the event. FileVault is on, so after a reboot the service only comes back once someone logs in.
- Keep a few GB of disk space free.
- Optional: run `sudo pmset -a disablesleep 1` to stop the Mac sleeping even with the lid closed. Run `sudo pmset -a disablesleep 0` to undo it.
- Make one test call from the website before each presentation.

## 4. Run it on LiveKit Cloud instead (no laptop needed)

`Dockerfile` and `livekit.toml` are set up for LiveKit Cloud agent hosting (agent `CA_eB73vdSi3Wc7`, region `us-east`).

The free Build plan includes **1,000 agent minutes a month**. Calls fail when that runs out, and you're never charged. After the agent has been idle, the first call can take 10–20 seconds to connect.

```bash
brew install livekit-cli
```

```bash
cd ~/Zoey/SoulDating
set -a; source .env; set +a
lk agent status
```

`AGENT_NAME` controls who can reach the cloud copy:

- **Empty or unset:** it answers website calls, like the Mac service.
- **`AGENT_NAME=zoey-cloud-test` (current setting):** test-only. Website calls never reach it.

To make the cloud copy answer website calls, upload the secrets. That restarts the agent. `AGENT_NAME=` is left empty on purpose, which turns test-only mode off:

```bash
lk agent update-secrets \
  --secrets DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY \
  --secrets GROQ_API_KEY=$GROQ_API_KEY \
  --secrets ZOEY_INGEST_URL=$ZOEY_INGEST_URL \
  --secrets ZOEY_INGEST_SECRET=$ZOEY_INGEST_SECRET \
  --secrets AGENT_NAME=
```

Deploy a new version after changing code:

```bash
lk agent deploy
```

Watch the cloud logs:

```bash
lk agent logs
```

If the Mac service and the cloud copy both answer website calls, LiveKit sends each call to one of them. That's safe, so you can switch over without downtime.

## Where the data goes

- **Call profiles:** stored in a Vercel Blob store (`profiles/call-*.json`) and shown at `https://www.zoeyai.love/backend?key=<DASHBOARD_KEY>`. Click **↻ Refresh** after a call to see it. The page doesn't refresh on its own, which keeps Vercel's free-tier usage low.
- **Local copies:** saved in `logs/profiles/`. They're git-ignored because they contain real callers' data.
