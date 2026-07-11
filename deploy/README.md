# sortAI — Autostart Setup

This directory contains platform-specific files for running the sortAI
dashboard (with inbox watching) automatically on startup.

All methods log to `logs/dashboard.log` inside the project folder.

---

## Windows — System tray app

Runs the dashboard as a background app with a system tray icon — no console
window, no Task Scheduler. The tray menu has **Open Dashboard** and **Quit**.

**1. Install the tray extras** (into the project venv):

```bat
.venv\Scripts\activate
pip install -e ".[tray]"
```

**2. Install the autostart shortcut** (and start immediately):

```bat
powershell -ExecutionPolicy Bypass -File deploy\windows\install-autostart.ps1 -StartNow
```

This creates *sortAI Dashboard.lnk* in your Startup folder (`shell:startup`)
pointing at `.venv\Scripts\sortai-tray.exe` with the project root as working
directory — no placeholders to edit. The tray app starts at every logon.

**3. Verify**

Look for the sortAI icon in the system tray (you may need to expand the
overflow area). Click it (or choose **Open Dashboard**) to open
`http://localhost:8765`. Logs go to `logs\dashboard.log` (rotating, 10 MB × 3
files). If sortAI is already running, a second launch shows a notice and just
opens the dashboard in the browser.

**Uninstall:**

```bat
powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-autostart.ps1
```

This removes the Startup shortcut; quit a running instance via the tray
menu's **Quit**.

---

## Linux — systemd user service

The unit redirects stdout/stderr to `logs/dashboard.log`; alternatively, add
`--log-file logs/dashboard.log` to the `sortai dashboard` command for built-in
rotating file logging (10 MB × 3 files).

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

The agent redirects stdout/stderr to `logs/dashboard.log`; alternatively, add
`--log-file logs/dashboard.log` to the `sortai dashboard` command for built-in
rotating file logging (10 MB × 3 files).

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
