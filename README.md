# NSE volume surge scanner

Scans NIFTY 500 every minute during market hours (09:15–15:30 IST) and flags stocks
trading on abnormally high volume versus their 20-day average, time-of-day adjusted.
Runs on GitHub Actions in two sessions a day; alerts go to Telegram; each run's
alerts CSV is attached to the workflow run under "Artifacts".

Setup: Settings → Secrets and variables → Actions → add `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`. Then Actions → "Test NSE access" → Run workflow.
