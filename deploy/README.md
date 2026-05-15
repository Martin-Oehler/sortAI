# sortAI — Autostart Setup

This directory contains platform-specific configuration files for running
`sortai dashboard --watch --no-browser` automatically on startup.

All methods log stdout/stderr to `logs/dashboard.log` inside the project folder.

---

## Windows — Task Scheduler (recommended)

Runs at user logon. No extra tools required.

**1. Edit the path placeholder**

Open `windows/sortai-task.xml` and replace both occurrences of `SORTAI_ROOT`
with the absolute path to your project folder, e.g.
`C:\Users\you\Documents\git\sortAI`.

**2. Import the task**

```bat
schtasks /Create /XML deploy\windows\sortai-task.xml /TN "sortai-dashboard"
```

Or open **Task Scheduler** → *Action* → *Import Task…* and select the XML file.

**3. Verify**

Right-click *sortai-dashboard* in Task Scheduler and choose **Run**.
Check `logs\dashboard.log` for the uvicorn startup line, then browse to
`http://localhost:8765`.

**Manage:**

```bat
schtasks /Run    /TN "sortai-dashboard"   :: start now
schtasks /End    /TN "sortai-dashboard"   :: stop
schtasks /Delete /TN "sortai-dashboard"   :: remove
```

---

## Windows — NSSM Service (headless / multi-user machines)

Use this when sortai must run even when no user is logged in.
Requires [NSSM](https://nssm.cc/):

```bat
winget install nssm
```

Run the installer **as Administrator** from the project root:

```bat
deploy\windows\install-nssm-service.bat
```

The script auto-detects the venv, installs the service as `sortai-dashboard`,
enables log rotation (10 MB), and starts it immediately.

**Manage:**

```bat
nssm start  sortai-dashboard
nssm stop   sortai-dashboard
nssm remove sortai-dashboard confirm
```

---

## Linux — systemd user service

**1. Edit the path placeholder**

Open `linux/sortai.service` and replace `/home/youruser/sortAI` with the
absolute path to your project folder.

**2. Install and enable**

```bash
mkdir -p ~/.config/systemd/user
cp deploy/linux/sortai.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sortai
```

**3. Verify**

```bash
systemctl --user status sortai
# or follow the log:
tail -f logs/dashboard.log
```

**Manage:**

```bash
systemctl --user start   sortai
systemctl --user stop    sortai
systemctl --user restart sortai
systemctl --user disable sortai
```

> To run as a system service (survives without login), copy the unit to
> `/etc/systemd/system/`, add `User=youruser` under `[Service]`, and use
> `systemctl` without `--user`.

---

## macOS — launchd LaunchAgent

**1. Edit the path placeholder**

Open `macos/com.sortai.dashboard.plist` and replace `/Users/youruser/sortAI`
(three occurrences) with the absolute path to your project folder.

**2. Install and load**

```bash
cp deploy/macos/com.sortai.dashboard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.sortai.dashboard.plist
```

The agent starts immediately and again at every login (`RunAtLoad + KeepAlive`).

**3. Verify**

```bash
launchctl list | grep sortai
tail -f logs/dashboard.log
```

**Manage:**

```bash
launchctl start  com.sortai.dashboard
launchctl stop   com.sortai.dashboard
launchctl unload ~/Library/LaunchAgents/com.sortai.dashboard.plist
```
